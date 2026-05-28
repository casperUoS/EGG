# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
from typing import List, Optional

import numpy
import numpy as np
from scipy import stats
from timm.utils import accuracy
from torch import nn, optim
from numpy.linalg import norm
from torchvision import models

from gensim.models import Word2Vec, KeyedVectors

try:
    # requires python >= 3.7
    from contextlib import nullcontext
except ImportError:
    # not exactly the same, but will do for our purposes
    from contextlib import suppress as nullcontext

import torch
from torch.utils.data import DataLoader, Dataset

from .batch import Batch
from .callbacks import (
    Callback,
    Checkpoint,
    CheckpointSaver,
    ConsoleLogger,
    TensorboardLogger,
)
from .distributed import get_preemptive_checkpoint_dir
from .interaction import Interaction
from .util import get_opts, move_to

try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    pass

class SketchDataset(Dataset):
    def __init__(self, messages, labels):
        """
        :param messages: Tensor of shape (num_samples, height, width, channels)
        """
        self.messages = messages
        self.labels = labels

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        message = self.messages[idx]
        label = self.labels[idx]
        return message, label


class Trainer:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """

    def __init__(
        self,
        game: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_data: DataLoader,
        optimizer_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        validation_data: Optional[DataLoader] = None,
        device: torch.device = None,
        callbacks: Optional[List[Callback]] = None,
        grad_norm: float = None,
        aggregate_interaction_logs: bool = True,
        run = None,
        vgg_path = "",
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param optimizer_scheduler: An optimizer scheduler to adjust lr throughout training
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer = optimizer
        self.optimizer_scheduler = optimizer_scheduler
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.run = run

        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks if callbacks else []
        self.grad_norm = grad_norm
        self.aggregate_interaction_logs = aggregate_interaction_logs

        self.update_freq = common_opts.update_freq

        self.vgg_path = vgg_path
        self.full_interaction = None

        if common_opts.load_from_checkpoint is not None:
            print(
                f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}"
            )
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        self.distributed_context = common_opts.distributed_context
        if self.distributed_context.is_distributed:
            print("# Distributed context: ", self.distributed_context)

        if self.distributed_context.is_leader and not any(
            isinstance(x, CheckpointSaver) for x in self.callbacks
        ):
            if common_opts.preemptable:
                assert (
                    common_opts.checkpoint_dir
                ), "checkpointing directory has to be specified"
                d = get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
                self.checkpoint_path = d
                self.load_from_latest(d)
            else:
                self.checkpoint_path = (
                    None
                    if common_opts.checkpoint_dir is None
                    else pathlib.Path(common_opts.checkpoint_dir)
                )

            if self.checkpoint_path:
                checkpointer = CheckpointSaver(
                    checkpoint_path=self.checkpoint_path,
                    checkpoint_freq=common_opts.checkpoint_freq,
                )
                self.callbacks.append(checkpointer)

        if self.distributed_context.is_leader and common_opts.tensorboard:
            assert (
                common_opts.tensorboard_dir
            ), "tensorboard directory has to be specified"
            tensorboard_logger = TensorboardLogger()
            self.callbacks.append(tensorboard_logger)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

        if self.distributed_context.is_distributed:
            device_id = self.distributed_context.local_rank
            torch.cuda.set_device(device_id)
            self.game.to(device_id)

            # NB: here we are doing something that is a bit shady:
            # 1/ optimizer was created outside of the Trainer instance, so we don't really know
            #    what parameters it optimizes. If it holds something what is not within the Game instance
            #    then it will not participate in distributed training
            # 2/ if optimizer only holds a subset of Game parameters, it works, but somewhat non-documentedly.
            #    In fact, optimizer would hold parameters of non-DistributedDataParallel version of the Game. The
            #    forward/backward calls, however, would happen on the DistributedDataParallel wrapper.
            #    This wrapper would sync gradients of the underlying tensors - which are the ones that optimizer
            #    holds itself.  As a result it seems to work, but only because DDP doesn't take any tensor ownership.

            self.game = torch.nn.parallel.DistributedDataParallel(
                self.game,
                device_ids=[device_id],
                output_device=device_id,
                find_unused_parameters=True,
            )
            self.optimizer.state = move_to(self.optimizer.state, device_id)

        else:
            self.game.to(self.device)
            # NB: some optimizers pre-allocate buffers before actually doing any steps
            # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
            # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
            self.optimizer.state = move_to(self.optimizer.state, self.device)

        if common_opts.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def eval(self, data=None):
        mean_loss = 0.0
        interactions = []
        n_batches = 0
        validation_data = self.validation_data if data is None else data
        self.game.eval()
        with torch.no_grad():
            for batch in validation_data:
                if not isinstance(batch, Batch):
                    batch = Batch(*batch)
                batch = batch.to(self.device)
                optimized_loss, interaction = self.game(*batch)
                if (
                    self.distributed_context.is_distributed
                    and self.aggregate_interaction_logs
                ):
                    interaction = Interaction.gather_distributed_interactions(
                        interaction
                    )
                interaction = interaction.to("cpu")
                mean_loss += optimized_loss

                for callback in self.callbacks:
                    callback.on_batch_end(
                        interaction, optimized_loss, n_batches, is_training=False
                    )

                interactions.append(interaction)
                n_batches += 1

        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)

        self.full_interaction = full_interaction

        return mean_loss.item(), full_interaction

    def semantic_correlaton(self, vgg, val_dataloader):

        cate_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

        word2vec = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

        cate2vec = {}
        for name in cate_names:
            cate2vec[name] = numpy.asarray(word2vec[name], dtype=np.float32)

        feature_extractor = vgg.features

        cate_features = {}

        with torch.no_grad():
            for batch_id, batch in enumerate(val_dataloader):
                inputs, labels = batch
                features = feature_extractor(inputs)

                for i, label_idx in enumerate(labels):
                    label_name = cate_names[label_idx.item()]
                    if label_name not in cate_features:
                        cate_features[label_name] = []
                    cate_features[label_name].append(features[i].cpu())

        cate2feature = {}
        for name in cate_names:
            cate_feature = cate_features[name]
            cate_feature = numpy.array(cate_feature).squeeze()
            cate_feature = cate_feature.mean(axis=0)
            cate2feature[name] = cate_feature

        x = []
        y = []
        for cate_i in cate_names:
            for cate_j in cate_names:
                vec1 = cate2vec[cate_i]
                vec2 = cate2vec[cate_j]
                sim = vec1.dot(vec2) / (norm(vec1) * norm(vec2))
                x.append(sim)

                vec1 = cate2feature[cate_i]
                vec2 = cate2feature[cate_j]
                sim = vec1.dot(vec2) / (norm(vec1) * norm(vec2))
                y.append(sim)

        a = stats.pearsonr(np.array(x), np.array(y))
        return a[0]

    def symbolicity_eval(self, epochs = 10):

        messages = self.full_interaction.message
        messages = torch.squeeze(messages)
        messages = messages.unsqueeze(1)
        messages = messages.repeat(1, 3, 1, 1)
        ids = self.full_interaction.labels

        train_messages, val_messages = np.split(messages, [int(len(messages) * 0.9)])
        train_labels, val_labels = np.split(ids, [int(len(ids) * 0.9)])

        train_sketches = SketchDataset(train_messages,train_labels)
        test_sketches = SketchDataset(val_messages,val_labels)

        train_dataloader = DataLoader(train_sketches, batch_size=16, shuffle=True, num_workers=2)
        val_dataloader = DataLoader(test_sketches, batch_size=16, shuffle=False, num_workers=2)


        vgg = models.vgg16(pretrained=True)

        vgg.classifier[-1] = nn.Linear(vgg.classifier[-1].in_features, 10)

        for param in vgg.features.parameters():
            param.requires_grad = False

        for param in vgg.classifier[:-1].parameters():
            param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(vgg.parameters(), lr=0.0002, momentum=0.9)


        for epoch in range(epochs):
            print("epoch:", epoch)
            running_loss = 0.0
            for batch_id, batch in enumerate(train_dataloader):
                inputs, labels = batch

                optimizer.zero_grad()

                outputs = vgg(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                # if batch_id % 2000 == 1999:  # print every 2000 mini-batches
                #     print(f'[{epoch + 1}, {batch_id + 1:5d}] loss: {running_loss / 2000:.3f}')
                #     running_loss = 0.0
            print(f'[{epoch + 1}] loss: {running_loss / len(train_dataloader):.3f}')

        print("evaluating symbolicity ...")

        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_id, batch in enumerate(val_dataloader):
                inputs, labels = batch
                outputs = vgg(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                total += labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()

        eval_loss = running_loss / len(val_dataloader)
        eval_accuracy = (100 * correct / total)

        sem_col = self.semantic_correlaton(vgg, val_dataloader)

        return eval_loss , eval_accuracy, sem_col

    def train_epoch(self):
        mean_loss = 0
        n_batches = 0
        interactions = []

        self.game.train()

        self.optimizer.zero_grad()

        for batch_id, batch in enumerate(self.train_data):
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(self.device)

            context = autocast() if self.scaler else nullcontext()
            with context:
                optimized_loss, interaction = self.game(*batch)

                if self.update_freq > 1:
                    # throughout EGG, we minimize _mean_ loss, not sum
                    # hence, we need to account for that when aggregating grads
                    optimized_loss = optimized_loss / self.update_freq

            if self.scaler:
                self.scaler.scale(optimized_loss).backward()
            else:
                optimized_loss.backward()

            if batch_id % self.update_freq == self.update_freq - 1:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                if self.grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.game.parameters(), self.grad_norm
                    )
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            n_batches += 1
            mean_loss += optimized_loss.detach()
            if (
                self.distributed_context.is_distributed
                and self.aggregate_interaction_logs
            ):
                interaction = Interaction.gather_distributed_interactions(interaction)
            interaction = interaction.to("cpu")

            for callback in self.callbacks:
                callback.on_batch_end(interaction, optimized_loss, batch_id)

            interactions.append(interaction)

        if self.optimizer_scheduler:
            self.optimizer_scheduler.step()

        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)
        return mean_loss.item(), full_interaction

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch + 1)

            train_loss, train_interaction = self.train_epoch()
            self.run.log({"train_loss": train_loss})
            if "acc" in train_interaction.aux:
                self.run.log({"train_accuracy": train_interaction.aux["acc"].mean()})
            if "baseline" in train_interaction.aux:
                self.run.log({"train_baseline": train_interaction.aux["baseline"].mean()})
            if "receiver_entropy" in train_interaction.aux:
                self.run.log({"train_receiver_entropy": train_interaction.aux["receiver_entropy"].mean()})
            self.run.log({"epoch": epoch})

            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_interaction, epoch + 1)

            validation_loss = validation_interaction = None
            if (
                self.validation_data is not None
                and self.validation_freq > 0
                and (epoch + 1) % self.validation_freq == 0
            ):
                for callback in self.callbacks:
                    callback.on_validation_begin(epoch + 1)
                validation_loss, validation_interaction = self.eval()
                self.run.log({"test_loss": validation_loss})
                if "acc" in validation_interaction.aux:
                    self.run.log({"test_accuracy": validation_interaction.aux["acc"].mean()})
                if "baseline" in validation_interaction.aux:
                    self.run.log({"test_baseline": validation_interaction.aux["baseline"].mean()})
                if "receiver_entropy" in validation_interaction.aux:
                    self.run.log({"test_receiver_entropy": validation_interaction.aux["receiver_entropy"].mean()})

                for callback in self.callbacks:
                    callback.on_validation_end(
                        validation_loss, validation_interaction, epoch + 1
                    )

            if self.should_stop:
                for callback in self.callbacks:
                    callback.on_early_stopping(
                        train_loss,
                        train_interaction,
                        epoch + 1,
                        validation_loss,
                        validation_interaction,
                    )
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        if checkpoint.optimizer_scheduler_state_dict:
            self.optimizer_scheduler.load_state_dict(
                checkpoint.optimizer_scheduler_state_dict
            )
        self.start_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """7
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f"# loading trainer state from {path}")
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob("*.tar"):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
import torch.nn.functional as F
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import egg.core as core
from egg.zoo.signal_game.archs import InformedSender, Receiver
from egg.zoo.signal_game_drawing.features import ImageNetFeat, ImagenetLoader, CIFAR10WithObj2ID
from egg.zoo.signal_game_drawing.archs import DrawSender, DrawReceiver, DrawReceiverClassifier
from egg.zoo.signal_game_drawing.wrappers import BezierReinforceWrapper
import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_root", default="", help="data root folder")
    # 2-agents specific parameters
    parser.add_argument(
        "--tau_s", type=float, default=10.0, help="Sender Gibbs temperature"
    )
    parser.add_argument(
        "--game_size", type=int, default=2, help="Number of images seen by an agent"
    )
    parser.add_argument("--same", type=int, default=0, help="Use same concepts")
    parser.add_argument("--embedding_size", type=int, default=50, help="embedding size")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=20,
        help="hidden size (number of filters informed sender)",
    )
    parser.add_argument(
        "--batches_per_epoch",
        type=int,
        default=100,
        help="Batches in a single training/validation epoch",
    )
    parser.add_argument("--inf_rec", type=int, default=0, help="Use informed receiver")
    parser.add_argument(
        "--mode",
        type=str,
        default="rf",
        help="Training mode: Gumbel-Softmax (gs) or Reinforce (rf). Default: rf.",
    )
    parser.add_argument("--gs_tau", type=float, default=1.0, help="GS temperature")
    parser.add_argument("--sample_mode", default="all", help="'all': display all classes. 'single' display one class, 'double' display two classes")
    parser.add_argument("--all_classes", type=bool, default=False, help="Turns signal game into classification game")

    opt = core.init(parser)
    assert opt.game_size >= 1

    return opt


def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    """
    Accuracy loss - non-differetiable hence cannot be used with GS
    """
    acc = (labels == receiver_output).float()
    return -acc, {"acc": acc}


def loss_nll(
    _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
):
    """
    NLL loss - differentiable and can be used with both GS and Reinforce
    """
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float().mean()
    return nll, {"acc": acc}


def get_game(config):
    feat_size = 512
    sender = DrawSender(
        feat_size=feat_size,
        vgg_path=opts.vgg_root,
        signal_game=False if config['all_classes'] else True,
    )
    if config["all_classes"]:
        receiver = DrawReceiverClassifier(
            vgg_path=opts.vgg_root,
        )
    else:
        receiver = DrawReceiver(
            game_size=config['game_size'],
            feat_size=feat_size,
            vgg_path=opts.vgg_root,
        )
    if opts.mode == "rf":
        sender = BezierReinforceWrapper(sender, config['canvas_size'])
        receiver = core.ReinforceWrapper(receiver)
        game = core.SymbolGameDrawReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=config['sender_entropy_coeff'],
            receiver_entropy_coeff=config['receiver_entropy_coeff'],
        )
    elif opts.mode == "gs":
        sender = core.GumbelSoftmaxWrapper(sender, temperature=opts.gs_tau)
        game = core.SymbolGameGS(sender, receiver, loss_nll)
    else:
        raise RuntimeError(f"Unknown training mode: {opts.mode}")

    return game


if __name__ == "__main__":

    wandb.login()

    opts = parse_arguments()

    if opts.all_classes:
        project = "REINFORCE Sketch Classification Game"
    else:
        project = "REINFORCE Sketch Lewis Game"

    config = dict(
        epochs=opts.n_epochs,
        classes=10,
        batch_size=opts.batch_size,
        batches_per_epoch=opts.batches_per_epoch,
        learning_rate=opts.lr,
        dataset='cifar10',
        game_size=opts.game_size,
        sender_entropy_coeff=0.0000001,
        receiver_entropy_coeff=0.1,
        all_classes=opts.all_classes,
        canvas_size=32,
        same_vgg_model=True
    )

    # data_folder = os.path.join(opts.root, "train/")
    cifar_path = "data/cifar10"
    dataset_exists = os.path.exists(os.path.join(cifar_path, "cifar-10-batches-py"))
    # dataset = ImageNetFeat(root=data_folder)

    with wandb.init(project=project, config=config) as run:



        if config["all_classes"]:
            train_dataset = CIFAR10(root=cifar_path, train=True, download=not dataset_exists, transform=transforms.ToTensor())
            test_dataset = CIFAR10(root=cifar_path, train=False, download=not dataset_exists, transform=transforms.ToTensor())
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
            validation_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        else:
            train_dataset = CIFAR10WithObj2ID(cifar_path, train=True, download=not dataset_exists)
            test_dataset = CIFAR10WithObj2ID(cifar_path, train=False, download=not dataset_exists)
            train_loader = ImagenetLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                opt=opts,
                batches_per_epoch=config['batches_per_epoch'],
                seed=None,
            )
            validation_loader = ImagenetLoader(
                test_dataset,
                opt=opts,
                batch_size=config['batch_size'],
                batches_per_epoch=config['batches_per_epoch'],
                seed=21,
            )
        game = get_game(config)
        optimizer = core.build_optimizer(game.parameters())
        callback = None
        if opts.mode == "gs":
            callbacks = [core.TemperatureUpdater(agent=game.sender, decay=0.9, minimum=0.1)]
        else:
            callbacks = []

        callbacks.append(core.ConsoleLogger(as_json=True, print_train_loss=True))
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=validation_loader,
            callbacks=callbacks,
            run=run
        )

        trainer.train(n_epochs=config['epochs'])

        # 1. Run inference on the validation set to get a batch of interactions
        print("Generating sample sketch...")
        val_loss, interaction = trainer.eval()

        for sample_mode in ["all","single","double"]:

            sketches = interaction.message.detach().cpu()
            splines = interaction.sender_output.detach().cpu()
            sender_input = interaction.sender_input.detach().cpu()
            # receiver_input = interaction.receiver_input.detach().cpu()
            receiver_output = interaction.receiver_output.detach().cpu()
            labels = interaction.labels.detach().cpu()
            edge_penalty = interaction.edge_penalty.detach().cpu()

            # 3. Plot and save one sample
            import matplotlib.pyplot as plt

            # Pick the first image in the batch
            sample = sketches[0]

            # Remove the channel dimension if it exists (e.g., convert 1x28x28 to 28x28)
            if sample.ndim == 3:
                sample = sample.squeeze(0)

            # print("sample =", splines[0])
            # print(splines.shape)
            #
            # print("sender_input =", sender_input[0][0])
            # print("sender_shape=", sender_input.shape )
            # print("reciever_input =", reciever_input[0][0])
            # print("reciever_shape=", reciever_input.shape )

            # print("reciever_output=",receiver_output)
            # print("labels=",labels)

            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']

            num_samples = min(32, sketches.size(0))
            max_rows = 8
            num_cols = ((num_samples - 1) // max_rows) + 1  # Calculate number of column pairs needed
            num_rows = min(num_samples, max_rows)

            # Create a grid: rows x (2 * num_cols) since each sample needs 2 subplots
            fig, axes = plt.subplots(num_rows, 2 * num_cols, figsize=(8 * num_cols, num_rows * 3))

            # Handle single column case
            if num_cols == 1 and num_rows == 1:
                axes = axes.reshape(1, -1)
            elif num_cols == 1:
                axes = axes.reshape(num_rows, -1)
            elif num_rows == 1:
                axes = axes.reshape(-1, 2 * num_cols)



            # print("single_class_idx", len(single_class_idx))
            # print("single_class_idx", single_class_idx)

            if sample_mode == "single":
                single_class_idx = (labels == 0).nonzero(as_tuple=True)[0]
                sketches = sketches[single_class_idx]
                labels = labels[single_class_idx]
                edge_penalty = edge_penalty[single_class_idx]
                if sender_input.ndim == 5:
                    sender_input = sender_input.index_select(dim=1, index=single_class_idx)
                else:
                    sender_input = sender_input[single_class_idx]

            if sample_mode == "double":
                class1_idx = (labels == 0).nonzero(as_tuple=True)[0][0:(num_samples//2)]
                class2_idx = (labels == 5).nonzero(as_tuple=True)[0][0:(num_samples//2)]
                sketches = torch.cat((sketches[class1_idx], sketches[class2_idx]))
                labels = torch.cat([labels[class1_idx], labels[class2_idx]])
                edge_penalty = torch.cat([edge_penalty[class1_idx], edge_penalty[class2_idx]])
                if sender_input.ndim == 5:
                    sender_input = torch.cat([sender_input.index_select(dim=1, index=class1_idx), sender_input.index_select(dim=1, index=class2_idx)], dim=1)
                else:
                    sender_input = torch.cat([sender_input[class1_idx], sender_input[class2_idx]])

            for i in range(num_samples):
                # Calculate which column pair and row this sample belongs to
                col_pair = i // max_rows
                row = i % max_rows

                sketch_sample = sketches[i]

                if sender_input.ndim == 5:
                    original_sample = sender_input[0, i]
                else:
                    original_sample = sender_input[i]

                if sketch_sample.ndim == 3:
                    sketch_sample = sketch_sample.squeeze(0)

                if original_sample.ndim == 3:
                    original_sample = original_sample.permute(1, 2, 0)

                if original_sample.max() > 1.0:
                    original_sample = original_sample / 255.0

                class_idx = labels[i].item()
                class_name = class_names[class_idx]
                edge_penalty_sample = edge_penalty[i]

                # Calculate column indices for original and sketch
                orig_col = col_pair * 2
                sketch_col = col_pair * 2 + 1

                # Plot original image
                axes[row, orig_col].imshow(original_sample)
                axes[row, orig_col].set_title(f"Original: {class_name}")
                axes[row, orig_col].axis('off')

                # Plot sketch
                axes[row, sketch_col].imshow(sketch_sample, cmap='gray', origin='lower')
                axes[row, sketch_col].set_title(f"Sketch: {class_name}, edge_penalty: {edge_penalty_sample}")
                axes[row, sketch_col].axis('off')

            # Hide unused subplots
            for i in range(num_samples, num_rows * num_cols):
                col_pair = i // max_rows
                row = i % max_rows
                axes[row, col_pair * 2].axis('off')
                axes[row, col_pair * 2 + 1].axis('off')

            plt.tight_layout()
            plt.suptitle("Original Images vs Sketches from Trained Sender", y=1.00)
            wandb.log({f"plot_{sample_mode}": fig})

        # plt.show()

        core.close()

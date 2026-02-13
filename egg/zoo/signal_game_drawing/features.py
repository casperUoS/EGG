# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data as data
from torchvision import transforms


class _BatchIterator:
    def __init__(self, loader, n_batches, seed=None, diff_class=False):
        self.loader = loader
        self.n_batches = n_batches
        self.batches_generated = 0
        self.random_state = np.random.RandomState(seed)
        self.diff_class = diff_class

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_generated > self.n_batches:
            raise StopIteration()

        batch_data = self.get_batch()
        self.batches_generated += 1
        return batch_data

    def get_batch(self):
        loader = self.loader
        opt = loader.opt

        C = len(self.loader.dataset.obj2id.keys())  # number of concepts

        if self.diff_class:
            game_size_sender = opt.game_size + 1
        else:
            game_size_sender = opt.game_size


        images_indexes_sender = np.zeros((opt.batch_size, game_size_sender))

        for b in range(opt.batch_size):
            if opt.same:
                # randomly sample a concept
                concepts = self.random_state.choice(C, 1)
                c = concepts[0]
                ims = loader.dataset.obj2id[c]["ims"]
                idxs_sender = self.random_state.choice(
                    ims, opt.game_size, replace=False
                )
                images_indexes_sender[b, :] = idxs_sender
            else:
                idxs_sender = []
                # randomly sample k concepts
                concepts = self.random_state.choice(C, opt.game_size, replace=False)
                if self.diff_class:
                    sample_num = 2
                else:
                    sample_num = 1
                for i, c in enumerate(concepts):
                    ims = loader.dataset.obj2id[c]["ims"]
                    idx = self.random_state.choice(ims, sample_num, replace=False)
                    idxs_sender.extend(idx)
                    sample_num = 1

                images_indexes_sender[b, :] = np.array(idxs_sender)

        images_vectors_sender = []

        # print(images_indexes_sender)
        for i in range(game_size_sender):
            batch_indices = images_indexes_sender[:, i]

            batch_images = []
            for b_idx in batch_indices:
                img, _ = loader.dataset[int(b_idx)]
                batch_images.append(img)

            # Stack the list of images into a single tensor
            images_vectors_sender.append(torch.stack(batch_images))

        images_vectors_sender = torch.stack(images_vectors_sender).contiguous()

        y_sender = torch.zeros(opt.batch_size).long()
        for i in range(opt.batch_size):
            target_image_idx = int(images_indexes_sender[i, 0])
            y_sender[i] = loader.dataset.targets[target_image_idx]

        y_reciever = torch.zeros(opt.batch_size).long()
        if self.diff_class:
            images_vectors_sender= images_vectors_sender[1:]
        images_vectors_receiver = torch.zeros_like(images_vectors_sender)
        for i in range(opt.batch_size):
            permutation = torch.randperm(opt.game_size)

            images_vectors_receiver[:, i, :] = images_vectors_sender[permutation, i, :]
            y_reciever[i] = permutation.argmin()
        return images_vectors_sender, y_sender, images_vectors_receiver, y_reciever


class ImagenetLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        self.opt = kwargs.pop("opt")
        self.seed = kwargs.pop("seed")
        self.batches_per_epoch = kwargs.pop("batches_per_epoch")
        self.diff_class = kwargs.pop("diff_class")

        super(ImagenetLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        if self.seed is None:

            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed
        return _BatchIterator(self, n_batches=self.batches_per_epoch, seed=seed, diff_class=self.diff_class)



class CIFAR10WithObj2ID(data.Dataset):
    def __init__(self, root, train=True, download=False):
        from torchvision.datasets import CIFAR10

        self.dataset = CIFAR10(root=root, train=train, download=download, transform=transforms.ToTensor())
        self.data = self.dataset.data
        self.targets = self.dataset.targets
        self.create_obj2id()

    def create_obj2id(self):
        """Create mapping from class labels to image indices"""
        self.obj2id = {}

        for class_idx in range(10):  # CIFAR-10 has 10 classes
            self.obj2id[class_idx] = {
                "labels": class_idx,
                "ims": []
            }

        # Populate with image indices for each class
        # Use self.targets instead of iterating over self.data
        for i, label in enumerate(self.targets):
            self.obj2id[label]["ims"].append(i)

    def __getitem__(self, index):
        img, _target = self.dataset[index]
        return img, index

    def __len__(self):
        return len(self.dataset)


class ImageNetFeat(data.Dataset):
    def __init__(self, root, train=True):
        import h5py

        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        # FC features
        fc_file = os.path.join(root, "ours_images_single_sm0.h5")

        fc = h5py.File(fc_file, "r")
        # There should be only 1 key
        key = list(fc.keys())[0]
        # Get the data
        data = torch.FloatTensor(list(fc[key]))

        # normalise data
        img_norm = torch.norm(data, p=2, dim=1, keepdim=True)
        normed_data = data / img_norm

        objects_file = os.path.join(root, "ours_images_single_sm0.objects")
        with open(objects_file, "rb") as f:
            labels = pickle.load(f)
        objects_file = os.path.join(root, "ours_images_paths_sm0.objects")
        with open(objects_file, "rb") as f:
            paths = pickle.load(f)

        self.create_obj2id(labels)
        self.data_tensor = normed_data
        self.labels = labels
        self.paths = paths

    def __getitem__(self, index):
        return self.data_tensor[index], index

    def __len__(self):
        return self.data_tensor.size(0)

    def create_obj2id(self, labels):
        self.obj2id = {}
        keys = {}
        idx_label = -1
        for i in range(labels.shape[0]):
            if not labels[i] in keys.keys():
                idx_label += 1
                keys[labels[i]] = idx_label
                self.obj2id[idx_label] = {}
                self.obj2id[idx_label]["labels"] = labels[i]
                self.obj2id[idx_label]["ims"] = []
            self.obj2id[idx_label]["ims"].append(i)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.fx.experimental.unification.utils import freeze
from torch.nn import Flatten
import numpy as np

import torchvision.models as models

from egg.zoo import signal_game


# class model_VGG16(nn.Module):
#     def __init__(self, num_classes):
#         super(model_VGG16, self).__init__()
#         # Load pre-trained VGG16 model
#         vgg16_model = models.vgg16(weights='IMAGENET1K_V1')
#
#         # Modify the first convolutional layer to accept the specified number of input channels
#         self.features = vgg16_model.features
#         #         self.features[0] = nn.Conv2d(num_input_channels, 64, kernel_size=3, padding=1)
#
#         # Additional layers similar to the ResNet50 model
#         self.additional_layers = nn.Sequential(
#             nn.Flatten(),
#             nn.BatchNorm1d(512),  # 128 -> 512
#             nn.Linear(512, 512),  # 128x128 -> 512x512
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.BatchNorm1d(512),  # 128 -> 512
#             nn.Linear(512, num_classes),  # 128 -> 512
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.additional_layers(x)
#         return x

class DrawSender(nn.Module):
    # This was mainly taken from dani's learning to draw model
    def __init__(
            self,
            feat_size,
            vgg_path,
            hidden_size = 512,
            num_splines = 3,
            signal_game = True
    ):
        super(DrawSender,self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.signal_game = signal_game


        # cifar10_weights = torch.load("/home/casper/Documents/Data/cifar10vgg_data/cifar10vgg.h5")

        self.vgg = torch.load(vgg_path, weights_only=False)

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.lin1 = nn.Linear(feat_size, hidden_size, bias=True)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.lin2 = nn.Linear(hidden_size, 6*num_splines, bias=True)


    def forward(self, x, state=None):
        if self.signal_game:
            x = x[0]

        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.lin1(x))
        # x = self.bn1(x)
        x = self.lin2(x)
        return x

class DrawReceiver(nn.Module):
    def __init__(self, game_size, feat_size, vgg_path, dropout_rate=0.4, action_dim=2, embedding_size=50, freeze_vgg=True):
        super(DrawReceiver, self).__init__()

        self.game_size = game_size

        self.lin1 = nn.Linear(feat_size, embedding_size, bias=True)
        # self.lin2 = nn.Embedding(vocab_size, embedding_size)

        # self.vgg16 = models.vgg16(pretrained=True)
        # if freeze_vgg:
        #     for p in self.vgg16.parameters():
        #         p.requires_grad = False

        # cifar10_weights = torch.load("/home/casper/Documents/Data/cifar10vgg_data/cifar10vgg.h5")

        self.vgg = torch.load(vgg_path, weights_only=False)

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.enc = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            Flatten(),
            nn.Linear(8 * 8 * 256, embedding_size)
        )

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1) ,bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), bias=True)

        self.dense1 = nn.Linear(in_features=30976, out_features=1024, bias=True)
        self.dense2 = nn.Linear(in_features=1024, out_features=256, bias=True)
        self.denseFinal = nn.Linear(in_features=256, out_features=embedding_size, bias=True)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, signal, x, _aux_input=None):
        # embed each image (left or right)
        emb = self.return_embeddings(x)
        # embed the signal
        # if len(signal.size()) == 3:
        #     signal = signal.unsqueeze(1)
        # h_s = self.conv1(signal)
        # h_s = F.relu(h_s)
        # h_s = self.conv2(h_s)
        # h_s = F.relu(h_s)
        # h_s = self.conv3(h_s)
        # h_s = F.relu(h_s)
        # h_s = h_s.reshape((h_s.shape[0], -1))  # Flatten
        #
        # # Embedding Layer
        # emb_s = self.dense1(h_s)
        # embd_s = F.relu(emb_s)
        # embd_s = self.dropout(embd_s)
        # embd_s = self.dense2(embd_s)
        # embd_s = F.relu(embd_s)
        # embd_s = self.dropout(embd_s)
        # embd_s = self.denseFinal(embd_s)
        if len(signal.size()) == 3:
            signal = signal.unsqueeze(1)
            signal = signal.expand(-1,3,-1,-1)
        h_s = self.vgg(signal)
        if len(h_s.size()) == 3:
            h = h_s.squeeze(dim=-1)
        h_s = h_s.view(h_s.size(0), -1)
        h_s = self.lin1(h_s)
        # embd_s is of size batch_size x embedding_size
        h_s = h_s.unsqueeze(dim=1)
        # h_s is of size batch_size x 1 x embedding_size
        h_s = h_s.transpose(1, 2)
        # h_s is of size batch_size x embedding_size x 1
        out = torch.bmm(emb, h_s)
        # out is of size batch_size x game_size x 1
        out = out.squeeze(dim=-1)
        # out is of size batch_size x game_size
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def return_embeddings(self, x):
        # embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            h = self.vgg(h)
            if len(h.size()) == 3:
                h = h.squeeze(dim=-1)
            h = h.view(h.size(0), -1)
            h_i = self.lin1(h)
            # h_i are batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)
            # h_i are now batch_size x 1 x embedding_size
            embs.append(h_i)
        h = torch.cat(embs, dim=1)
        return h

class DrawReceiverClassifier(nn.Module):
    def __init__(self, vgg_path, dropout_rate=0.4, action_dim=2, classes=10, freeze_vgg=True):
        super(DrawReceiverClassifier, self).__init__()

        self.vgg = torch.load(vgg_path, weights_only=False)

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1) ,bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), bias=True)

        self.dense1 = nn.Linear(in_features=30976, out_features=1024, bias=True)
        self.dense2 = nn.Linear(in_features=1024, out_features=256, bias=True)
        self.denseFinal = nn.Linear(in_features=256, out_features=classes, bias=True)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, signal, x, _aux_input=None):
        # embed each image (left or right)
        # embed the signal
        if len(signal.size()) == 3:
            signal = signal.unsqueeze(1)
        h_s = self.conv1(signal)
        h_s = F.relu(h_s)
        h_s = self.conv2(h_s)
        h_s = F.relu(h_s)
        h_s = self.conv3(h_s)
        h_s = F.relu(h_s)
        h_s = h_s.reshape((h_s.shape[0], -1))  # Flatten

        # Embedding Layer
        emb_s = self.dense1(h_s)
        embd_s = F.relu(emb_s)
        embd_s = self.dropout(embd_s)
        embd_s = self.dense2(embd_s)
        embd_s = F.relu(embd_s)
        embd_s = self.dropout(embd_s)
        out = self.denseFinal(embd_s)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
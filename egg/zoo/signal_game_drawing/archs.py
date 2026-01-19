import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Flatten
import numpy as np

import torchvision.models as models

class DrawSender(nn.Module):
    # This was mainly taken from dani's learning to draw model
    def __init__(
            self,
            feat_size,
            hidden_size = 512,
            num_splines = 3
    ):
        super(DrawSender,self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size

        self.lin1 = nn.Linear(feat_size, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.lin2 = nn.Linear(hidden_size, 7*num_splines, bias=False)


    def forward(self, x, state=None):
        x = x[0] #get target image
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        return torch.sigmoid(x)

class DrawReceiver(nn.Module):
    def __init__(self, game_size, feat_size, dropout_rate=0.4, action_dim=2, embedding_size=50, freeze_vgg=True):
        super(DrawReceiver, self).__init__()

        self.game_size = game_size

        self.lin1 = nn.Linear(feat_size, embedding_size, bias=False)
        # self.lin2 = nn.Embedding(vocab_size, embedding_size)

        # self.vgg16 = models.vgg16(pretrained=True)
        # if freeze_vgg:
        #     for p in self.vgg16.parameters():
        #         p.requires_grad = False

        self.enc = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            Flatten(),
            nn.Linear(8 * 8 * 256, embedding_size)
        )

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(3, 3), bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(3, 3), bias=False)

        self.dense1 = nn.Linear(in_features=576, out_features=128, bias=False)
        self.dense2 = nn.Linear(in_features=128, out_features=128, bias=False)
        self.denseFinal = nn.Linear(in_features=128, out_features=embedding_size, bias=False)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, signal, x, _aux_input=None):
        # embed each image (left or right)
        emb = self.return_embeddings(x)
        # embed the signal
        if len(signal.size()) == 3:
            signal = signal.unsqueeze(1)
        h_s = self.conv1(signal)
        h_s = F.relu(h_s)
        h_s = self.conv2(h_s)
        h_s = F.relu(h_s)
        h_s = h_s.reshape((h_s.shape[0], -1))  # Flatten

        # Embedding Layer
        emb_s = self.dense1(h_s)
        embd_s = F.relu(emb_s)
        embd_s = self.dropout(embd_s)
        embd_s = self.dense2(embd_s)
        embd_s = F.relu(embd_s)
        embd_s = self.dropout(embd_s)
        embd_s = self.denseFinal(embd_s)
        # embd_s is of size batch_size x embedding_size
        embd_s = embd_s.unsqueeze(dim=1)
        # h_s is of size batch_size x 1 x embedding_size
        emdb_s = embd_s.transpose(1, 2)
        # h_s is of size batch_size x embedding_size x 1
        out = torch.bmm(emb, emdb_s)
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
            if len(h.size()) == 3:
                h = h.squeeze(dim=-1)
            h_i = self.lin1(h)
            # h_i are batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)
            # h_i are now batch_size x 1 x embedding_size
            embs.append(h_i)
        h = torch.cat(embs, dim=1)
        return h
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Flatten
import numpy as np

import torchvision.models as models

canvas_shape = (28, 28)

def paint_multiple_splines(all_spline_params):
    """Paint multiple splines on a single canvas."""

    def paint_spline_on_canvas(spline_params):
        """Paint a single spline on the canvas with specified thickness using advanced indexing."""

        def bezier_spline(t, P0, P1, P2):
            """Compute points on a quadratic Bézier spline for a given t."""
            t = t[:, None]  # Shape (N, 1) to broadcast with P0, P1, P2 of shape (2,)
            P = (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2
            return P  # Returns shape (N, 2), a list of points on the spline

        brush_size = 0

        spline_params *= canvas_shape[0]

        canvas = np.zeros(canvas_shape)

        for spline in spline_params:

            # P0, P1, P2 = spline_params.reshape((3, 2))
            P0, P1, P2, W = spline[0:2], spline[2:4], spline[4:6], spline[6]
            W *= -0.003  # This is the weight param. -0.005 is too dark. -0.002 may be too light.
            t_values = np.linspace(0, 1, num=50)
            spline_points = bezier_spline(t_values, P0, P1, P2)
            x_points, y_points = np.round(spline_points).astype(int).T

            # Generate brush offsets
            brush_offsets = np.array([(dx, dy) for dx in range(-brush_size, brush_size + 1)  # brush_size + 1
                                       for dy in range(-brush_size, brush_size + 1)])  # brush_size + 1
            x_offsets, y_offsets = brush_offsets.T

            # Calculate all indices to update for each point (broadcasting magic)
            all_x_indices = x_points[:, None] + x_offsets
            all_y_indices = y_points[:, None] + y_offsets

            canvas[all_x_indices, all_y_indices] += W
        return canvas

    background_shade = 0.3  # This is the background color! For nearly all experiments it has been 0.2*number of splines. For a larger sig gap go for 0.1*number of splines

    splines = []

    for spline_params in all_spline_params:
        spline_params = paint_spline_on_canvas(spline_params.reshape(-1, 7))
        splines.append(spline_params)
    canvas = np.clip(np.array(splines) + background_shade, 0.0, 1.0)

    # print(all_spline_params.reshape(-1, 7))

    # all_spline_params = np.clip(all_spline_params, 0.0, 1.0)
    # canvas = np.clip(paint_spline_on_canvas(all_spline_params.reshape(-1, 7)).sum(axis=0) + background_shade, 0.0, 1.0)
    return canvas



class DrawSender(nn.Module):
    # This was mainly taken from dani's learning to draw model
    def __init__(
            self,
            feat_size,
            hidden_size,
            freeze_vgg = True,
            latent = 7*3
    ):
        super(DrawSender,self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size

        self.vgg16 = models.vgg16(pretrained=True)
        if freeze_vgg:
            for p in self.vgg16.parameters():
                p.requires_grad = False

        self.enc = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            Flatten(),
            nn.Linear(8 * 8 * 256, latent)
        )


    def forward(self, x, state=None):
        x = self.vgg16(x)
        x = self.enc(x)
        return x

class DrawReceiver(nn.Module):
    def __init__(self, game_size, dropout_rate=0.4, action_dim=2, embedding_size=50, freeze_vgg=True):
        super(DrawReceiver, self).__init__()

        self.game_size = game_size

        self.vgg16 = models.vgg16(pretrained=True)
        if freeze_vgg:
            for p in self.vgg16.parameters():
                p.requires_grad = False

        self.enc = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            Flatten(),
            nn.Linear(8 * 8 * 256, embedding_size)
        )

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same')

        self.dense1 = nn.Linear(in_features=256, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=128)
        self.denseFinal = nn.Linear(in_features=128, out_features=50)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, signal, x, _aux_input=None):
        # embed each image (left or right)
        emb = self.return_embeddings(x)
        # embed the signal
        if len(signal.size()) == 3:
            signal = signal.squeeze(dim=-1)
        h_s = self.conv1(signal)
        h_s = nn.ReLU(h_s)
        h_s = self.cvon2(h_s)
        h_s = nn.ReLU(h_s)
        h_s = h_s.reshape((h_s.shape[0], -1))  # Flatten

        # Embedding Layer
        emb_s = self.dense1(h_s)
        embd_s = nn.ReLU(emb_s)
        embd_s = self.dropout(embd_s)
        embd_s = self.dense2(embd_s)
        embd_s = nn.ReLU(embd_s)
        embd_s = self.dropout(embd_s)
        embd_s = self.denseFinal(embd_s)
        embd_s = nn.ReLU(embd_s)


    def return_embeddings(self, x):
        # embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            if len(h.size()) == 3:
                h = h.squeeze(dim=-1)
            h_i = self.vgg16(h)
            h_i = self.enc(h_i)
            # h_i are batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)
            # h_i are now batch_size x 1 x embedding_size
            embs.append(h_i)
        h = torch.cat(embs, dim=1)
        return h
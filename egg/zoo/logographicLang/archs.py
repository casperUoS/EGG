import torch
from torch import nn
import torch.nn.functional as F

import pydiffvg


class VisionEncoder(nn.Module):

    def __init__(self, feat_size, vision_path, hidden_size, z_dim=20, num_splines=3,
                 critic_mode=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_size = feat_size
        self.hidden_size = hidden_size

        self.vision = torch.load(vision_path, weights_only=False)

        for param in self.vision.parameters():
            param.requires_grad = False
        self.vision.eval()

        self.lin1 = nn.Sequential(nn.Linear(feat_size, hidden_size, bias=True), nn.LeakyReLU())

        self.fc_mu = nn.Linear(hidden_size, z_dim, bias=True)
        self.fc_logvar = nn.Linear(hidden_size, z_dim, bias=True)

        self.signal_game = True

        # self.logvar_predictor = nn.Linear(256 * 1 * 1, out_features)
        # self.mu_predictor = nn.Linear(256 * 1 * 1, out_features)

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()  # always keep vision_model in eval mode

        return self


    def forward(self, x):
        # if self.signal_game:
        #     x = x[0]

        embeds = self.vision(x)
        x = embeds.view(embeds.size(0), -1)
        x = self.lin1(x)

        logvar = self.fc_logvar(x)
        mu = self.fc_mu(x)

        auxdata = {
            "mu": mu,
            "logvar": logvar,
            "embeds": embeds,
        }

        return mu, logvar, auxdata

class DiffDecoder(nn.Module):

    def __init__(self, canvas_size=28, zdim=20, hdim=1024, paths=3, segments=1):
        super(DiffDecoder, self).__init__()
        self.imsize = canvas_size

        self.paths = paths
        self.segments = segments
        self.stroke_width = (2.0, 2.0)

        self.decoder = nn.Sequential(
            nn.Linear(zdim, hdim),
            nn.SELU(inplace=True),

            nn.Linear(hdim, hdim),
            nn.SELU(inplace=True),
        )

        self.point_predictor = nn.Sequential(
            nn.Linear(hdim, 2 * self.paths * (self.segments * 3 + 1)),
            nn.Tanh()  # bound spatial extent
        )

    def render(self, canvas_width, canvas_height, shapes, shape_groups, samples=2):
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups)
        img = _render(canvas_width,
                      canvas_height,
                      samples,
                      samples,
                      0,
                      None,
                      *scene_args)
        return img

    def decode(self, z):
        bs = z.shape[0]

        feats = self.decoder(z)

        all_points = self.point_predictor(feats)
        all_points = all_points.view(bs, self.paths, -1, 2)

        all_points = all_points * (self.imsize // 2 - 2) + self.imsize // 2

        # Process the batch sequentially
        outputs = []
        scenes = []
        for k in range(bs):
            # Get point parameters from network
            shapes = []
            shape_groups = []
            for p in range(self.paths):
                points = all_points[k, p].contiguous().cpu()
                width = torch.tensor(1.0).cpu()
                alpha = torch.tensor(1.0).cpu()

                color = torch.cat([torch.ones(3), alpha.view(1, )])
                num_ctrl_pts = torch.zeros(self.segments, dtype=torch.int32) + 2

                path = pydiffvg.Path(
                    num_control_points=num_ctrl_pts, points=points,
                    stroke_width=width, is_closed=False)

                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=None,
                    stroke_color=color)
                shape_groups.append(path_group)

            scenes.append(
                [shapes, shape_groups, (self.imsize, self.imsize)])

            # Rasterize
            out = self.render(self.imsize, self.imsize, shapes, shape_groups,
                         samples=1)

            # Torch format, discard alpha, make gray
            out = out.permute(2, 0, 1).view(
                4, self.imsize, self.imsize)[:3].mean(0, keepdim=True)

            outputs.append(out)

        output = torch.stack(outputs).to(z.device)

        auxdata = {
            "points": all_points,
            "scenes": scenes,
        }

        # map to [-1, 1]
        # output = output * 2.0 - 1.0

        output = output.squeeze()

        return output, auxdata

    def forward(self, z):
        output, aux = self.decode(z)
        return output, aux

class SketchEncoder(nn.Module):
    def __init__(self, dropout_rate=0.4, action_dim=2, classes=10, freeze_vgg=True):
        super(SketchEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), bias=True)

        self.dense1 = nn.Linear(in_features=30976, out_features=1024, bias=True)
        self.dense2 = nn.Linear(in_features=1024, out_features=256, bias=True)
        self.denseFinal = nn.Linear(in_features=256, out_features=classes, bias=True)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        if len(x.size()) == 3:
            signal = x.unsqueeze(1)
        h_s = self.conv1(x)
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
        h_s = self.denseFinal(embd_s)

        auxdata = {
            "embedding": embd_s,
        }

        return h_s, auxdata
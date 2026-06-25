import torch
from torch import nn

import pydiffvg


class VisionEncoder(nn.Module):

    def __init__(self, feat_size, vision_path, hidden_size, z_dim=128, num_splines=3,
                 critic_mode=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_size = feat_size
        self.hidden_size = hidden_size

        self.vision = torch.load(vision_path, weights_only=False)

        for param in self.vision.parameters():
            param.requires_grad = False
        self.vision.eval()

        self.fc_mu = nn.Linear(feat_size, z_dim, bias=True)
        self.fc_logvar = nn.Linear(feat_size, z_dim, bias=True)

        # self.logvar_predictor = nn.Linear(256 * 1 * 1, out_features)
        # self.mu_predictor = nn.Linear(256 * 1 * 1, out_features)

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()  # always keep vision_model in eval mode

        return self


    def forward(self, x):
        embeds = self.vision(x)
        x = embeds.view(embeds.size(0), -1)

        logvar = self.fc_mu(x)
        mu = self.fc_logvar(x)

        auxdata = {
            "mu": mu,
            "logvar": logvar,
            "embeds": embeds,
        }

        return mu, logvar, auxdata

class DiffDencoder(nn.Module):

    def __init__(self, agent, canvas_size=28, zdim=128, hdim=1024, paths=3, segments=1):
        super(DiffDencoder, self).__init__()
        self.agent = agent
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

        # self.width_predictor = nn.Sequential(
        #     nn.Linear(hdim, self.paths),
        #     nn.Sigmoid()
        # )

        # self.alpha_predictor = nn.Sequential(
        #     nn.Linear(hdim, self.paths),
        #     nn.Sigmoid()
        # )

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

        # all_widths = self.width_predictor(z)
        # min_width = self.stroke_width[0]
        # max_width = self.stroke_width[1]
        # all_widths = (max_width - min_width) * all_widths + min_width

        # all_alphas = self.alpha_predictor(z)

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
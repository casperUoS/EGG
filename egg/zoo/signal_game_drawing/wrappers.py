import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import Normal, Independent

import pydiffvg


class DiffRasterWrapper(nn.Module):

    def __init__(self, agent, canvas_size=28, zdim=20, hdim=1024, paths=3, segments=1):
        super(DiffRasterWrapper, self).__init__()
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(logvar)
        return mu + std*eps, std

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

    def forward(self, *args, **kwargs):

        mu, logvar, vgg_feats = self.agent(*args, **kwargs)
        z,std = self.reparameterize(mu, logvar)

        output, aux = self.decode(z)

        aux["sender_log_prob"] = logvar
        aux["sender_entropy"] = std.mean()
        aux["vgg_features"] = vgg_feats
        return output, aux




class BezierReinforceWrapper(nn.Module):
    """
    Reinforce Wrapper for an agent. Assumes that the during the forward,
    the wrapped agent returns log-probabilities over the potential outputs. During training, the wrapper
    transforms them into a tuple of (sample from the multinomial, log-prob of the sample, entropy for the multinomial).
    Eval-time the sample is replaced with argmax.

    >>> agent = nn.Sequential(nn.Linear(10, 3), nn.LogSoftmax(dim=1))
    >>> agent = ReinforceWrapper(agent)
    >>> sample, log_prob, entropy = agent(torch.ones(4, 10))
    >>> sample.size()
    torch.Size([4])
    >>> (log_prob < 0).all().item()
    1
    >>> (entropy > 0).all().item()
    1
    """

    def __init__(self, agent, canvas_size=28, std=0.1):
        super(BezierReinforceWrapper, self).__init__()
        self.agent = agent
        self.log_std = nn.Parameter(torch.ones(1) * np.log(std))
        self.canvas_size = canvas_size


    def paint_multiple_splines(self, all_spline_samples):

        device = all_spline_samples.device
        batch_size = all_spline_samples.size(0)
        num_t = 50

        # all_spline_samples: (batch, numsplines * 7)
        params = all_spline_samples.view(batch_size, -1, 6) * self.canvas_size
        # params: (batch, num_splines, 7)

        # P1, P2, P3: (batch, splines, 2)
        P0, P1, P2, = params[..., 0:2], params[..., 2:4], params[..., 4:6]
        # W: (batch, splines, 1)
        # W = params[..., 6:7] lmao
        num_splines = params.size(1)
        # W = 1
        # W = W * -0.003
        W = torch.full((batch_size, num_splines, 1), -0.07, device=device)

        t = torch.linspace(0, 1, steps=num_t, device=device).view(1, 1, num_t, 1)

        P0 = P0.unsqueeze(2)
        P1 = P1.unsqueeze(2)
        P2 = P2.unsqueeze(2)

        spline_points = (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2
        coords = torch.round(spline_points).long()

        brush_size = 1  # Adjustable brush size

        # Create offsets [-brush_size, ..., brush_size]
        r = torch.arange(-brush_size, brush_size + 1, device=device)
        dx, dy = torch.meshgrid(r, r, indexing='ij')
        dx, dy = dx.flatten(), dy.flatten()  # Shape: (num_offsets,)

        # Extract base coordinates: (Batch, Splines, num_t)
        base_x = coords[..., 0]
        base_y = coords[..., 1]

        # Add offsets: Broadcast (Batch, Splines, num_t, 1) + (num_offsets,)
        # Result: (Batch, Splines, num_t, num_offsets)
        x_indices = base_x.unsqueeze(-1) + dx
        y_indices = base_y.unsqueeze(-1) + dy

        # Clamp to canvas boundaries
        x_indices = torch.clamp(x_indices, 0, self.canvas_size - 1)
        y_indices = torch.clamp(y_indices, 0, self.canvas_size - 1)

        # Flatten spatial dims to 1D for scatter
        # Shape: (Batch, Total_Points) where Total_Points = Splines * num_t * num_offsets
        flat_x = x_indices.view(batch_size, -1)
        flat_y = y_indices.view(batch_size, -1)

        flat_indices = flat_x * self.canvas_size + flat_y

        # Expand weights to match the brush offsets dimensions
        # W starts as (Batch, Splines, 1), needs to match (Batch, Splines, num_t, num_offsets)
        num_offsets = dx.size(0)
        flat_weights = W.unsqueeze(-1).expand(-1, -1, num_t, num_offsets).reshape(batch_size, -1)

        canvas_flat = torch.zeros(batch_size, self.canvas_size * self.canvas_size, device=device)
        canvas_flat.scatter_add_(1, flat_indices, flat_weights)

        background_shade = 0.3
        canvas = canvas_flat.view(batch_size, self.canvas_size, self.canvas_size)
        canvas = torch.clamp(canvas + background_shade, 0.0, 1.0)

        return canvas

    def forward(self, *args, **kwargs):
        mu, vgg_features = self.agent(*args, **kwargs)

        std = self.log_std.exp()

        # dim = mu.size(-1)
        # scale_tril = torch.eye(dim, device=mu.device) * self.noise_std

        distr = Normal(loc=mu, scale=0.0001)
        distr = Independent(distr, 1)

        entropy = distr.entropy()

        if self.training:
            raw_sample = distr.sample()
        else:
            raw_sample = mu

        sample = torch.sigmoid(raw_sample)

        log_prob = distr.log_prob(raw_sample)

        sketch = self.paint_multiple_splines(sample)

        return sketch, log_prob, entropy, sample, vgg_features
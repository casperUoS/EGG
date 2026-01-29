import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import Normal, Independent




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
        mu = self.agent(*args, **kwargs)

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

        return sketch, log_prob, entropy, sample
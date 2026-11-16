import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Flatten
import numpy as np

import torchvision.models as models

canvas_shape = (28, 28)


def paint_multiple_splines_tensor( all_spline_samples):
    device = all_spline_samples.device
    batch_size = all_spline_samples.size(0)
    num_t = 50

    # all_spline_samples: (batch, numsplines * 7)
    params = all_spline_samples.view(batch_size, -1, 7) * canvas_shape[0]
    # params: (batch, num_splines, 7)

    # P1, P2, P3: (batch, splines, 2)
    P0, P1, P2, = params[..., 0:2], params[..., 2:4], params[..., 4:6]
    # W: (batch, splines, 1)
    W = params[..., 6:7]
    W = W * -0.003

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
    x_indices = torch.clamp(x_indices, 0, canvas_shape[0] - 1)
    y_indices = torch.clamp(y_indices, 0, canvas_shape[1] - 1)

    # Flatten spatial dims to 1D for scatter
    # Shape: (Batch, Total_Points) where Total_Points = Splines * num_t * num_offsets
    flat_x = x_indices.view(batch_size, -1)
    flat_y = y_indices.view(batch_size, -1)

    flat_indices = flat_x * canvas_shape[1] + flat_y

    # Expand weights to match the brush offsets dimensions
    # W starts as (Batch, Splines, 1), needs to match (Batch, Splines, num_t, num_offsets)
    num_offsets = dx.size(0)
    flat_weights = W.unsqueeze(-1).expand(-1, -1, num_t, num_offsets).reshape(batch_size, -1)

    canvas_flat = torch.zeros(batch_size, canvas_shape[0] * canvas_shape[1], device=device)
    canvas_flat.scatter_add_(1, flat_indices, flat_weights)

    background_shade = 0.3
    canvas = canvas_flat.view(batch_size, canvas_shape[0], canvas_shape[1])
    canvas = torch.clamp(canvas + background_shade, 0.0, 1.0)

    return canvas

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

splines = np.array([[0.37210482358932495, 0.5101042985916138, 0.535472571849823, 0.3512147068977356, 0.20872655510902405, 0.27592694759368896, 0.31, 0.5394357442855835, 0.6043564081192017, 0.4065988063812256, 0.5282517075538635, 0.6306172609329224, 0.3307557702064514, 0.32,0.6,0.2,0.3,0.2,0.8,0.9,0.6],
                    [0.5399, 0.2609, 0.7477, 0.4139, 0.7645, 0.3406, 0.3498, 0.4776, 0.8530,
                     0.5183, 0.2338, 0.6632, 0.7560, 0.5155, 0.6289, 0.5154, 0.4147, 0.5374,
                     0.1392, 0.2966, 0.4538],
                    [0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
                     0., 1., 1.]
                     ])
splines = torch.from_numpy(splines).float()

canvas = paint_multiple_splines_tensor(splines) # shape (num images, num splines * 6)

canvas.detach_()
canvas = canvas.numpy()

# canvas = paint_multiple_splines(splines)

# Visualization with matplotlib
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(canvas[1], cmap='Greys', origin='lower')
axs[0].set_title('Optimized Painted Splines on Canvas 1')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')

axs[1].imshow(canvas[2], cmap='viridis', origin='lower')
axs[1].set_title('Optimized Painted Splines on Canvas 2')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')

plt.tight_layout()
plt.show()



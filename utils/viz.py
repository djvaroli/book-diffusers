
import typing as t

import torch
from torchvision.utils import make_grid
import numpy as np
from numpy import typing as npt
import matplotlib.pyplot as plt



def plot_images_as_grid(
    images: npt.NDArray,
    nrow: int = 4,
    normalize: bool = True,
    padding: int = 2,
    pad_value: float = 0,
    figsize: t.Tuple[int, int] = (10, 12)
):
    """Given an array of images of shape (B, H, W, C) stacks them into a grid of shape (H1, W1, C) and
    then displays them.

    Args:
        images (npt.NDArray): images to display.
        nrow (int, optional): number of images per row. Defaults to 4.
        normalize (bool, optional): normalize values to be in [0, 1] range. Defaults to True.
        padding (int, optional): number of padding pixels between each image in grid. Defaults to 2.
        pad_value (float, optional): value at padding pixel locations. Defaults to 0.
        figsize (t.Tuple[int, int], optional): size of figure. Defaults to (10, 12).
    """
    # input images should have shape (B, H, W, C), we permute to fit Torch format of (B, C, H, W)
    images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
    
    # grid will have shape (C, H1, W1)
    grid = make_grid(
        images_tensor, 
        normalize=normalize,
        padding=padding,
        pad_value=pad_value
    )
    # convert to (H1, W1, C)
    grid_np = grid.permute(1, 2, 0).numpy()
    figure = plt.figure(figsize=figsize)
    plt.imshow(grid_np)
    

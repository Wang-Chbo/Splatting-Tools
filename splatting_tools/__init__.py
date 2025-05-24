#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
import math
from . import _C

def render_2d(viewpoint_camera, gaussians):
    
    mean_gaussians = gaussians.get_xyz

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    view_matrix = viewpoint_camera.world_view_transform
    proj_matrix = viewpoint_camera.full_proj_transform

    _W = viewpoint_camera.image_width
    _H = viewpoint_camera.image_height

    # [u, v, depth]
    uvd, _ = _C.get_gaussian_xy(mean_gaussians, view_matrix, proj_matrix, _H, _W)


    mask_2d = (torch.min(uvd, dim=1).values > 0.0 - 0.5) & \
                (uvd[:, 1] < _H - 0.5) & \
                (uvd[:, 0] < _W - 0.5)
    uv_2d_int = uvd[mask_2d, :2]  # 只保留有效的uv坐标

    # uvd转化为整数的像素坐标
    uv_2d_int = uv_2d_int.round().long()

    return uvd, uv_2d_int

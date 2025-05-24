/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor, torch::Tensor> 
getGaussianXy(
	torch::Tensor& means3D,							// ! gaussian xyz
	torch::Tensor& viewmatrix,						// ! 相机位姿变换
	torch::Tensor& projmatrix, 						// ! 投影变换
	const int image_height,
	const int image_width
);

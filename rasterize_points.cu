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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::tuple<torch::Tensor, torch::Tensor> getGaussianXy(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix, 
	const int image_height, 
	const int image_width)
{
	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;
	
	torch::Tensor xy = torch::full({P, 3}, 0.0, means3D.options().dtype(torch::kFloat32));				// ! [u, v, depth]
	torch::Tensor zero_tensor = torch::full({3, 2}, 0.0, means3D.options().dtype(torch::kFloat32));
	if(P!= 0)
	{
		CudaRasterizer::Rasterizer::getGaussianXy(
			P, 
			means3D.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(), 
			xy.contiguous().data<float>(), 
			W, H);
	}
	return std::make_tuple(xy, zero_tensor);
}

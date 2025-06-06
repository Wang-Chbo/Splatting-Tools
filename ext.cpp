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

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  // m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  // m.def("mark_visible", &markVisible);
  m.def("get_gaussian_xy", &getGaussianXy, "Get the pixel positions of Gaussians.");  // ! 补充函数，进行提取数据，可以简单地返回某一个shape，以检验代码的正确性

}
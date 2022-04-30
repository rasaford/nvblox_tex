/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include "nvblox/core/camera.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/image.h"
#include "nvblox/core/types.h"

// functions defined in this header file are included multiple times in .cu
// files accross the project. To still keep all function signatures unique, they
// have to be defined as inline.

namespace nvblox {

__device__ inline bool projectThreadVoxel(
    const Index3D* block_indices_device_ptr, const Camera& camera,
    const Transform& T_C_L, const float block_size, Eigen::Vector2f* u_px_ptr,
    float* u_depth_ptr);

__device__ inline bool projectThreadTexel(
    const Index3D* block_indices_device_ptr, const Camera& camera,
    const Transform& T_C_L, const float block_size, const Index2D& texel_idx,
    const TexVoxel::Dir dir, Vector2f* u_px_ptr, float* u_depth_ptr);

__global__ inline void checkBlocksInTruncationBand(
    const VoxelBlock<TsdfVoxel>** block_device_ptrs,
    const float truncation_distance_m,
    bool* contains_truncation_band_device_ptr);

__device__ inline Color blendTwoColors(const Color& first_color,
                                       float first_weight,
                                       const Color& second_color,
                                       float second_weight);

}  // namespace nvblox

#include "nvblox/integrators/cuda/impl/projective_integrators_common_impl.cuh"

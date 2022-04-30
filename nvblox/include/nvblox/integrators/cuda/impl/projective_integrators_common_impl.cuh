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

namespace nvblox {

__device__ inline bool projectThreadVoxel(
    const Index3D* block_indices_device_ptr, const Camera& camera,
    const Transform& T_C_L, const float block_size, Eigen::Vector2f* u_px_ptr,
    float* u_depth_ptr) {
  // The indices of the voxel this thread will work on
  // block_indices_device_ptr[blockIdx.x]:
  //                 - The index of the block we're working on (blockIdx.y/z
  //                   should be zero)
  // threadIdx.x/y/z - The indices of the voxel within the block (we
  //                   expect the threadBlockDims == voxelBlockDims)
  const Index3D block_idx = block_indices_device_ptr[blockIdx.x];
  const Index3D voxel_idx(threadIdx.z, threadIdx.y, threadIdx.x);

  // Voxel center point
  const Vector3f p_voxel_center_L = getCenterPostionFromBlockIndexAndVoxelIndex(
      block_size, block_idx, voxel_idx);
  // To camera frame
  const Vector3f p_voxel_center_C = T_C_L * p_voxel_center_L;

  // Project to image plane
  if (!camera.project(p_voxel_center_C, u_px_ptr)) {
    return false;
  }

  // Depth
  *u_depth_ptr = p_voxel_center_C.z();
  return true;
}

__device__ inline bool projectThreadTexel(
    const Index3D* block_indices_device_ptr, const Camera& camera,
    const Transform& T_C_L, const float block_size, const Index2D& texel_idx,
    const TexVoxel::Dir dir, Vector2f* u_px_ptr, float* u_depth_ptr) {
  const Index3D block_idx = block_indices_device_ptr[blockIdx.x];
  const Index3D voxel_idx(threadIdx.z, threadIdx.y, threadIdx.x);

  // Texel center point
  const Vector3f p_texel_center_L = getCenterPositionForTexel(
      block_size, block_idx, voxel_idx, texel_idx, dir);
  // To camera frame
  const Vector3f p_texel_center_C = T_C_L * p_texel_center_L;

  // Project to image plane
  if (!camera.project(p_texel_center_C, u_px_ptr)) {
    return false;
  }

  // Depth
  *u_depth_ptr = p_texel_center_C.z();
  return true;
}

__global__ inline void checkBlocksInTruncationBand(
    const VoxelBlock<TsdfVoxel>** block_device_ptrs,
    const float truncation_distance_m,
    bool* contains_truncation_band_device_ptr) {
  // A single thread in each block initializes the output to 0
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    contains_truncation_band_device_ptr[blockIdx.x] = 0;
  }
  __syncthreads();

  // Get the Voxel we'll check in this thread
  const TsdfVoxel voxel = block_device_ptrs[blockIdx.x]
                              ->voxels[threadIdx.z][threadIdx.y][threadIdx.x];

  // If this voxel in the truncation band, write the flag to say that the block
  // should be processed.
  // NOTE(alexmillane): There will be collision on write here. However, from my
  // reading, all threads' writes will result in a single write to global
  // memory. Because we only write a single value (1) it doesn't matter which
  // thread "wins".

  if (std::abs(voxel.distance) <= truncation_distance_m) {
    contains_truncation_band_device_ptr[blockIdx.x] = true;
  }
}

__device__ inline Color blendTwoColors(const Color& first_color,
                                       float first_weight,
                                       const Color& second_color,
                                       float second_weight) {
  float total_weight = first_weight + second_weight;

  first_weight /= total_weight;
  second_weight /= total_weight;

  Color new_color;
  new_color.r = static_cast<uint8_t>(std::round(
      first_color.r * first_weight + second_color.r * second_weight));
  new_color.g = static_cast<uint8_t>(std::round(
      first_color.g * first_weight + second_color.g * second_weight));
  new_color.b = static_cast<uint8_t>(std::round(
      first_color.b * first_weight + second_color.b * second_weight));

  return new_color;
}

}  // namespace nvblox

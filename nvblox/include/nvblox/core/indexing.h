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

#include "nvblox/core/types.h"
#include "nvblox/core/voxels.h"

namespace nvblox {

// Assuming a fixed-size voxel block, get the voxel index of a voxel at that
// position within a block.
__host__ __device__ inline Index3D getVoxelIndexFromPositionInLayer(
    const float block_size, const Vector3f& position);

__host__ __device__ inline Index3D getBlockIndexFromPositionInLayer(
    const float block_size, const Vector3f& position);

__host__ __device__ inline void getBlockAndVoxelIndexFromPositionInLayer(
    const float block_size, const Vector3f& position, Index3D* block_idx,
    Index3D* voxel_idx);

/// Gets the position of the minimum corner (i.e., the smallest towards negative
/// infinity of each axis).
__host__ __device__ inline Vector3f getPositionFromBlockIndexAndVoxelIndex(
    const float block_size, const Index3D& block_index,
    const Index3D& voxel_index);

__host__ __device__ inline Vector3f getPositionFromBlockIndex(
    const float block_size, const Index3D& block_index);

/// Gets the CENTER of the voxel.
__host__ __device__ inline Vector3f getCenterPostionFromBlockIndexAndVoxelIndex(
    const float block_size, const Index3D& block_index,
    const Index3D& voxel_index);

// pack a 3D voxel_index into a linear index. The linearization and
// unlinearization are a bijective mapping.
__host__ __device__ inline int linearizeVoxelIndex(const Index3D& voxel_index,
                                                   const int kVoxelsPerSide) {
  return kVoxelsPerSide * kVoxelsPerSide * voxel_index[0] +
         kVoxelsPerSide * voxel_index[1] + voxel_index[2];
}

__host__ __device__ inline Index3D unlinearizeVoxelIndex(
    const int linear_index, const int kVoxelsPerSide) {
  const int voxels_squared = kVoxelsPerSide * kVoxelsPerSide;
  return Index3D(linear_index / voxels_squared,
                 (linear_index % voxels_squared) / kVoxelsPerSide,
                 linear_index % kVoxelsPerSide);
}
/**
 * @brief Transfroms the texel_index to 2D texel coordinates
 * Where:
 *  - The point (0, 0) is the center of the texel plane
 *  - The width / height of the plane is the voxel size
 *
 * @param pixel_index
 * @param patch_width
 * @param voxel_size
 * @return __host__
 */
__host__ __device__ inline Vector2f getTexelCoordsfromIdx(
    const Index2D& pixel_index, const int patch_width, const float voxel_size);

__host__ __device__ inline Vector3f getCenterPositionForTexel(
    const float block_size, const Index3D& block_index,
    const Index3D& voxel_index, const Index2D& pixel_index,
    const TexVoxel::Dir dir);

/**
 * @brief Used for a mesh block's packed texture tiles
 *
 * @param uv_coords
 * @param patches_per_side
 * @return __host__
 */
__host__ __device__ inline Index2D getTileIndexFromUVs(
    const Vector2f& uv_coords, const int patches_per_side);

__host__ __device__ inline Vector2f getTopLeftUVFromTileIndex(
    const Index2D& tile_index, const int patches_per_side);

}  // namespace nvblox

#include "nvblox/core/impl/indexing_impl.h"

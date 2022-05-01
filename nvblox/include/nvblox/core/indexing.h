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

// Indexing functions for a VoxelLayer which does not have the concept of a
// "block". Take care not to mix up these functions with the ones defined above.

/**
 * @brief Get the Voxel Index from position in a **VoxelLayer**. This function
 * does not work for BlockLayers
 *
 * @param voxel_size
 * @param position
 * @return Voxel index
 */
__host__ __device__ inline Index3D getVoxelIndexFromPositionInVoxelLayer(
    const float voxel_size, const Vector3f& position);
/**
 * @brief Get the Position From a Voxel index in a **VoxelLayer** This function
 * does not work for BlockLayers
 *
 * @param voxel_size
 * @param voxel_index
 * @return position of the smallest corner (i.e. the smallest towards negative
 * infinity of each axis) of the Voxel
 */
__host__ __device__ inline Vector3f getPositionFromVoxelIndexInVoxelLayer(
    const float voxel_size, const Index3D& voxel_index);

/**
 * @brief Get the Position From a Voxel index in a **VoxelLayer** This function
 * does not work for BlockLayers
 *
 * @param voxel_size
 * @param voxel_index
 * @return position of the center of the Voxel
 */
__host__ __device__ inline Vector3f getCenterPositionFromVoxelIndexInVoxelLayer(
    const float voxel_size, const Index3D& voxel_index);

/**
 * @brief Transfroms the texel_index to 2D texel coordinates
 *
 * @param texel_index
 * @param voxels_per_side
 * @param voxel_size
 * @return Vector2f
 */
__host__ __device__ inline Vector2f getTexelCoordsfromIdx(
    const Index2D& texel_index, const int texels_per_side,
    const float voxel_size);

__host__ __device__ inline Vector3f getCenterPositionForTexel(
    const float block_size, const Index3D& block_index,
    const Index3D& voxel_index, const Index2D& texel_index,
    const TexVoxel::Dir dir);
}  // namespace nvblox

#include "nvblox/core/impl/indexing_impl.h"

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

#include <glog/logging.h>
#include <algorithm>

#include "nvblox/core/accessors.h"
#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

// Block accessors by index.
template <typename BlockType>
typename BlockType::Ptr BlockLayer<BlockType>::getBlockAtIndex(
    const Index3D& index) {
  // Look up the block in the hash?
  // And return it.
  auto it = blocks_.find(index);
  if (it != blocks_.end()) {
    return it->second;
  } else {
    return typename BlockType::Ptr();
  }
}

template <typename BlockType>
typename BlockType::ConstPtr BlockLayer<BlockType>::getBlockAtIndex(
    const Index3D& index) const {
  const auto it = blocks_.find(index);
  if (it != blocks_.end()) {
    return (it->second);
  } else {
    return typename BlockType::ConstPtr();
  }
}

template <typename BlockType>
typename BlockType::Ptr BlockLayer<BlockType>::allocateBlockAtIndex(
    const Index3D& index) {
  auto it = blocks_.find(index);
  if (it != blocks_.end()) {
    return it->second;
  } else {
    // Invalidate the GPU hash
    gpu_layer_view_up_to_date_ = false;
    // Blocks define their own method for allocation.
    auto insert_status =
        blocks_.emplace(index, BlockType::allocate(memory_type_));
    return insert_status.first->second;
  }
}

template <typename BlockType>
void BlockLayer<BlockType>::evictOldBlocks(
    const std::vector<Index3D>& block_indices) {
  // boundaries of an AxisAlignedBoundingBox of the given indices
  Index3D min(INT_MAX, INT_MAX, INT_MAX), max(INT_MIN, INT_MIN, INT_MIN);
  for (const Index3D& block_idx : block_indices) {
    min = min.cwiseMin(block_idx);
    max = max.cwiseMax(block_idx);
  }

  std::vector<Index3D> to_delete;
  int evicts = 0;
  for (const Index3D& block_idx : device_blocks_) {
    // evict the current index is not in the AABB
    if ((min.array() <= block_idx.array()).all() &&
        (block_idx.array() <= max.array()).all()) {
      continue;
    }
    auto iit = blocks_.find(block_idx);
    if (iit == blocks_.end()) {
      continue;
    }
    evicts++;
    blocks_[block_idx] = allocator_.toHost(iit->second);
    to_delete.push_back(block_idx);
  }
  for (const Index3D& idx : to_delete) {
    device_blocks_.erase(idx);
  }
}

template <typename BlockType>
void BlockLayer<BlockType>::prefetchBlocks(
    const std::vector<Index3D>& block_indices) {
  for (const Index3D& idx : block_indices) {
    auto it = blocks_.find(idx);
    typename BlockType::Ptr known_ptr;
    if (it != blocks_.end()) {
      known_ptr = it->second;
    } else {
      // The block has not been allocated --> invalidate the GPU Hash view
      gpu_layer_view_up_to_date_ = false;
    }
    blocks_[idx] = allocator_.toDevice(known_ptr);
    device_blocks_.emplace(idx);
  }
  std::cout << "prefetched " << block_indices.size() << " num device blocks "
            << device_blocks_.size() << " total blocks " << blocks_.size()
            << std::endl;
  allocator_.printUsage();
}

// Block accessors by position.
template <typename BlockType>
typename BlockType::Ptr BlockLayer<BlockType>::getBlockAtPosition(
    const Eigen::Vector3f& position) {
  return getBlockAtIndex(
      getBlockIndexFromPositionInLayer(block_size_, position));
}

template <typename BlockType>
typename BlockType::ConstPtr BlockLayer<BlockType>::getBlockAtPosition(
    const Eigen::Vector3f& position) const {
  return getBlockAtIndex(
      getBlockIndexFromPositionInLayer(block_size_, position));
}

template <typename BlockType>
typename BlockType::Ptr BlockLayer<BlockType>::allocateBlockAtPosition(
    const Eigen::Vector3f& position) {
  return allocateBlockAtIndex(
      getBlockIndexFromPositionInLayer(block_size_, position));
}

template <typename BlockType>
std::vector<Index3D> BlockLayer<BlockType>::getAllBlockIndices() const {
  std::vector<Index3D> indices;
  indices.reserve(blocks_.size());

  for (const auto& kv : blocks_) {
    indices.push_back(kv.first);
  }

  return indices;
}

template <typename BlockType>
bool BlockLayer<BlockType>::isBlockAllocated(const Index3D& index) const {
  const auto it = blocks_.find(index);
  return (it != blocks_.end());
}

template <typename BlockType>
typename BlockLayer<BlockType>::GPULayerViewType
BlockLayer<BlockType>::getGpuLayerView() const {
  if (!gpu_layer_view_) {
    gpu_layer_view_ = std::make_unique<GPULayerViewType>();
  }
  if (!gpu_layer_view_up_to_date_) {
    (*gpu_layer_view_).reset(const_cast<BlockLayer<BlockType>*>(this));
    gpu_layer_view_up_to_date_ = true;
  }
  return *gpu_layer_view_;
}

// VoxelBlockLayer

template <typename VoxelType>
void VoxelBlockLayer<VoxelType>::getVoxels(
    const std::vector<Vector3f>& positions_L,
    std::vector<VoxelType>* voxels_ptr,
    std::vector<bool>* success_flags_ptr) const {
  CHECK_NOTNULL(voxels_ptr);
  CHECK_NOTNULL(success_flags_ptr);

  cudaStream_t transfer_stream;
  checkCudaErrors(cudaStreamCreate(&transfer_stream));

  voxels_ptr->resize(positions_L.size());
  success_flags_ptr->resize(positions_L.size());

  for (int i = 0; i < positions_L.size(); i++) {
    const Vector3f& p_L = positions_L[i];
    // Get the block address
    Index3D block_idx;
    Index3D voxel_idx;
    getBlockAndVoxelIndexFromPositionInLayer(this->block_size_, p_L, &block_idx,
                                             &voxel_idx);
    const typename VoxelBlock<VoxelType>::ConstPtr block_ptr =
        this->getBlockAtIndex(block_idx);
    if (!block_ptr) {
      (*success_flags_ptr)[i] = false;
      continue;
    }
    (*success_flags_ptr)[i] = true;
    // Get the voxel address
    const auto block_raw_ptr = block_ptr.get();
    const VoxelType* voxel_ptr =
        &block_raw_ptr->voxels[voxel_idx.x()][voxel_idx.y()][voxel_idx.z()];
    // Copy the Voxel to the CPU
    cudaMemcpyAsync(&(*voxels_ptr)[i], voxel_ptr, sizeof(VoxelType),
                    cudaMemcpyDefault, transfer_stream);
  }
  cudaStreamSynchronize(transfer_stream);
  checkCudaErrors(cudaStreamDestroy(transfer_stream));
}

namespace internal {

template <typename is_voxel_layer>
inline float sizeArgumentFromVoxelSize(float voxel_size);

template <>
inline float sizeArgumentFromVoxelSize<std::true_type>(float voxel_size) {
  return voxel_size;
}

template <>
inline float sizeArgumentFromVoxelSize<std::false_type>(float voxel_size) {
  return voxel_size * VoxelBlock<bool>::kVoxelsPerSide;
}

}  // namespace internal

template <typename LayerType>
constexpr float sizeArgumentFromVoxelSize(float voxel_size) {
  return internal::sizeArgumentFromVoxelSize<
      typename traits::is_voxel_layer<LayerType>::type>(voxel_size);
}

}  // namespace nvblox

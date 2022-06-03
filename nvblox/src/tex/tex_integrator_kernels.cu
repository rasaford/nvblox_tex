#include <vector>

// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
#include "nvblox/core/common_names.h"
#include "nvblox/core/layer.h"
#include "nvblox/gpu_hash/cuda/gpu_hash_interface.cuh"
#include "nvblox/gpu_hash/cuda/gpu_indexing.cuh"
#include "nvblox/ray_tracing/sphere_tracer.h"
#include "nvblox/tex/tex_integrator_kernels.h"

namespace nvblox {
namespace tex {

/**
 * @brief Get the Tsdf value distance value at the current position
 *
 * @param gpu_hash_index_to_str
 * @param position
 * @param block_size
 * @param distance output distance. Only written to if the return value ==
 * true
 * @return if the given position corresponds to a valid voxel with weight > 0
 */
__device__ inline bool getTsdfDistance(
    const Index3DDeviceHashMapType<TsdfBlock>& gpu_hash_index_to_ptr,
    const Vector3f& position, const float block_size, float& distance) {
  TsdfVoxel* voxel;
  if (getVoxelAtPosition<TsdfVoxel>(gpu_hash_index_to_ptr, position, block_size,
                                    &voxel) &&
      voxel != nullptr) {
    if (voxel->weight > 0.0f) {
      distance = voxel->distance;
      return true;
    }
  }
  return false;
}

/**
 * @brief Computes TSDF gradient for the current position.
 * If a vertex does not have 6 neighbors, the gradient will be (0, 0, 0).
 *
 * @param gpu_hash_index_to_ptr
 * @param position
 * @param block_size
 * @param voxel_size
 * @param gradient gradient at the given position
 * @return if the computed gradient is valid (i.e. the voxel at this position
 * has at least 6 neighbors)
 */
__device__ bool computeTSDFGradient(
    const Index3DDeviceHashMapType<TsdfBlock>& gpu_hash_index_to_ptr,
    const Vector3f& position, const float block_size, const float voxel_size,
    Vector3f& gradient) {
  // voxel size is block size divided by number of blocks per side
  // const float voxel_size =
  //     block_size / static_cast<float>(tsdf_block->kVoxelsPerSide);
  const float voxel_size_2x = 2.0f * voxel_size;
  float dist_x_plus = 0, dist_x_minus = 0, dist_y_plus = 0, dist_y_minus = 0,
        dist_z_plus = 0, dist_z_minus = 0;
  bool valid = true;

  // get tsdf values for each neighboring voxel to the current one
  // clang-format off
  valid &= getTsdfDistance(gpu_hash_index_to_ptr, position + Vector3f(voxel_size, 0.0f, 0.0f), block_size, dist_x_plus); 
  valid &= getTsdfDistance(gpu_hash_index_to_ptr, position + Vector3f(0.0f, voxel_size, 0.0f), block_size, dist_y_plus); 
  valid &= getTsdfDistance(gpu_hash_index_to_ptr, position + Vector3f(0.0f, 0.0f, voxel_size), block_size, dist_z_plus); 
  valid &= getTsdfDistance(gpu_hash_index_to_ptr, position - Vector3f(voxel_size, 0.0f, 0.0f), block_size, dist_x_minus); 
  valid &= getTsdfDistance(gpu_hash_index_to_ptr, position - Vector3f(0.0f, voxel_size, 0.0f), block_size, dist_y_minus); 
  valid &= getTsdfDistance(gpu_hash_index_to_ptr, position - Vector3f(0.0f, 0.0f, voxel_size), block_size, dist_z_minus);
  // clang-format on

  if (!valid) {
    return false;
  }

  // approximate gradient by finite differences
  gradient << (dist_x_plus - dist_x_minus) / voxel_size_2x,
      (dist_y_plus - dist_y_minus) / voxel_size_2x,
      (dist_z_plus - dist_z_minus) / voxel_size_2x;
  return true;
}

/**
 * @brief quantizes the given direction vector (normalized) into one of
 * TexVoxel::Dir directions
 *
 * @param normal **normalized** direction vector
 * @return quantized direction
 */
__device__ inline TexVoxel::Dir quantizeDirection(const Vector3f& dir) {
  const Vector3f abs_dir = dir.cwiseAbs();
  TexVoxel::Dir res;
  if (abs_dir(0) >= abs_dir(1) && abs_dir(0) >= abs_dir(2)) {
    res = dir(0) < 0 ? TexVoxel::Dir::X_MINUS : res = TexVoxel::Dir::X_PLUS;
  } else if (abs_dir(1) >= abs_dir(2)) {
    res = dir(1) < 0 ? TexVoxel::Dir::Y_MINUS : TexVoxel::Dir::Y_PLUS;
  } else {
    res = dir(2) < 0 ? TexVoxel::Dir::Z_MINUS : TexVoxel::Dir::Z_PLUS;
  }
  return res;
}

/**
 * @brief Computes the weight (confidence) we have in the quantized direction
 * given the true gradient direction
 *
 * @param dir quantized direction of the given gradient
 * @param gradient **normalized** true gradient direction
 * @return __device__ weight \in [-1, 1], if dir is the most likely one: weight
 * \in [0, 1]
 */
__device__ inline float computeDirWeight(const TexVoxel::Dir dir,
                                         const Vector3f& gradient) {
  return texDirToWorldVector(dir).dot(gradient);
}

/**
 * @brief Updates the direction values for all TexVoxels where this has not
 * been set yet
 *
 * @param tsdf_block_ptrs
 * @param tex_block_ptrs
 * @param block_size
 */
__global__ void setTexVoxelDirsfromTsdfGradient(
    const Index3DDeviceHashMapType<TsdfBlock> gpu_hash_index_to_ptr,
    TexBlock** tex_block_ptrs, const Index3D* block_indices,
    const float block_size, const float voxel_size) {
  // Get the Voxels we'll check in this thread
  TexBlock* tex_block = tex_block_ptrs[blockIdx.x];
  const Index3D& block_index = block_indices[blockIdx.x];
  Index3D voxel_index = Index3D(threadIdx.z, threadIdx.y, threadIdx.x);
  TexVoxel* tex_voxel =
      &(tex_block->voxels[voxel_index[0]][voxel_index[1]][voxel_index[2]]);

  // only update the direction for voxels where we are not yet very confident in
  // thier direction
  // if (tex_voxel->dir_weight >= TexVoxel::DIR_THRESHOLD) return;

  Vector3f position = getCenterPostionFromBlockIndexAndVoxelIndex(
      block_size, block_index, voxel_index);

  // Since we are working in an TSDF, where the distance of each voxel to the
  // surface implicitly defines the surface boundary, the normal of each voxel
  // is just the normalized gradient.
  Vector3f gradient;
  const bool valid_gradient = computeTSDFGradient(
      gpu_hash_index_to_ptr, position, block_size, voxel_size, gradient);
  const double gradient_norm = gradient.norm();
  if (!valid_gradient || gradient_norm <= 0) return;

  gradient /= gradient_norm;  // normalize gradient inplace

  // Since the quanization of the normalized gradient to the 6 differnt major
  // axis directions in TexVoxel::Dir always introduces an error, we track the
  // confidence in the computed quantization. If we are sufficiently more
  // confident in a newly computed gradient direction we upate the associated
  // texture
  TexVoxel::Dir dir = quantizeDirection(gradient);
  float dir_weight = computeDirWeight(dir, gradient);
  if (TexVoxel::DIR_THRESHOLD * dir_weight >= tex_voxel->dir_weight) {
    // printf("new_dir_weight: %f, tex_voxel->dir_weight: %f\n", dir_weight,
    // tex_voxel->dir_weight);
    tex_voxel->updateDir(dir, dir_weight);
  }
  // printf("gradient: (%f %f %f), dir: %d\n", gradient[0], gradient[1],
  // gradient[2], tex_voxel.dir);
}

/**
 * @brief Computes a historgram where the index is the diretion of each
 * neighboring voxel and the value is the accumulated confidence for all voxels
 * with that direction
 *
 * @param gpu_hash_index_to_ptr
 * @param position
 * @param block_size
 * @param voxel_size
 * @param histogram
 * @return __device__ if the computed histogram is valid, i.e. there exist
 * neighboring voxels in all directions of the given position
 */
__device__ bool computeDirHistogram(
    const Index3DDeviceHashMapType<TexBlock>& gpu_hash_index_to_ptr,
    const Vector3f& position, const float block_size, const float voxel_size,
    Vector7f& weights, Vector7i& frequencies) {
  bool valid = true;

  // clang-format off
  TexVoxel *voxel;
  if (valid &= getVoxelAtPosition<TexVoxel>(gpu_hash_index_to_ptr, position                                 , block_size, &voxel)) { const auto idx = static_cast<int>(voxel->dir); frequencies[idx]++; weights[idx] += voxel->dir_weight; }
  if (valid &= getVoxelAtPosition<TexVoxel>(gpu_hash_index_to_ptr, position + Vector3f(voxel_size, 0.f, 0.f), block_size, &voxel)) { const auto idx = static_cast<int>(voxel->dir); frequencies[idx]++; weights[idx] += voxel->dir_weight; }
  if (valid &= getVoxelAtPosition<TexVoxel>(gpu_hash_index_to_ptr, position + Vector3f(0.f, voxel_size, 0.f), block_size, &voxel)) { const auto idx = static_cast<int>(voxel->dir); frequencies[idx]++; weights[idx] += voxel->dir_weight; }
  if (valid &= getVoxelAtPosition<TexVoxel>(gpu_hash_index_to_ptr, position + Vector3f(0.f, 0.f, voxel_size), block_size, &voxel)) { const auto idx = static_cast<int>(voxel->dir); frequencies[idx]++; weights[idx] += voxel->dir_weight; }
  if (valid &= getVoxelAtPosition<TexVoxel>(gpu_hash_index_to_ptr, position - Vector3f(voxel_size, 0.f, 0.f), block_size, &voxel)) { const auto idx = static_cast<int>(voxel->dir); frequencies[idx]++; weights[idx] += voxel->dir_weight; }
  if (valid &= getVoxelAtPosition<TexVoxel>(gpu_hash_index_to_ptr, position - Vector3f(0.f, voxel_size, 0.f), block_size, &voxel)) { const auto idx = static_cast<int>(voxel->dir); frequencies[idx]++; weights[idx] += voxel->dir_weight; }
  if (valid &= getVoxelAtPosition<TexVoxel>(gpu_hash_index_to_ptr, position - Vector3f(0.f, 0.f, voxel_size), block_size, &voxel)) { const auto idx = static_cast<int>(voxel->dir); frequencies[idx]++; weights[idx] += voxel->dir_weight; }
  // clang-format on
  return valid;
}

template <typename BlockType>
__device__ inline int linearizedThreadVoxelIdx() {
  constexpr int voxels_per_block = BlockType::kVoxelsPerSide *
                                   BlockType::kVoxelsPerSide *
                                   BlockType::kVoxelsPerSide;
  constexpr int voxels_per_slice =
      BlockType::kVoxelsPerSide * BlockType::kVoxelsPerSide;
  // clang-format off
  return blockIdx.x * voxels_per_block 
          + threadIdx.z * voxels_per_slice 
          + threadIdx.y * BlockType::kVoxelsPerSide 
          + threadIdx.x;
  // clang-format on
}

__global__ void majorityVoteTexVoxelDirs(
    const Index3DDeviceHashMapType<TexBlock> gpu_hash_index_to_ptr,
    TexBlock** tex_block_ptrs, const Index3D* block_indices,
    const float block_size, const float voxel_size, TexVoxel::Dir* smooth_dirs,
    float* smooth_weights) {
  // Get the Voxels we'll check in this thread
  const TexBlock* tex_block = tex_block_ptrs[blockIdx.x];
  const Index3D& block_index = block_indices[blockIdx.x];
  Index3D voxel_index = Index3D(threadIdx.z, threadIdx.y, threadIdx.x);
  const TexVoxel* tex_voxel =
      &(tex_block->voxels[voxel_index[0]][voxel_index[1]][voxel_index[2]]);

  Vector3f position = getCenterPostionFromBlockIndexAndVoxelIndex(
      block_size, block_index, voxel_index);

  Vector7f weights = Vector7f::Zero();
  Vector7i frequencies = Vector7i::Zero();
  const bool valid_hist =
      computeDirHistogram(gpu_hash_index_to_ptr, position, block_size,
                          voxel_size, weights, frequencies);

  // write current dir to smoothed output
  const int linear_voxel_idx = linearizedThreadVoxelIdx<TexBlock>();
  smooth_dirs[linear_voxel_idx] = tex_voxel->dir;
  smooth_weights[linear_voxel_idx] = tex_voxel->dir_weight;

  if (!valid_hist) return;

  // find highest weight and second highest to determine if we overwrite the
  // TexVoxel direction.
  // TODO(rasaford) this is a very inefficient way to do an array partition.
  // Replace this with a better impelmentation
  int max_idx = -1, second_idx = -1;
  float max = -1.f, second = -1.f;
  for (int i = 0; i < 7; ++i) {
    if (weights[i] > max) {
      max_idx = i;
      max = weights[i];
    }
  }
  for (int i = 0; i < 7; ++i) {
    if (weights[i] < max && weights[i] > second) {
      second_idx = i;
      second = weights[i];
    }
  }

  // if the most common direction is twice as likely as the second most common
  // one, we update the current voxels direction to be the most common one
  constexpr float DIR_UPDATE_CONFIDENCE = .5f;
  if (DIR_UPDATE_CONFIDENCE * max > second && max > 0 && second > 0) {
    smooth_dirs[linear_voxel_idx] = static_cast<TexVoxel::Dir>(max_idx);
    smooth_weights[linear_voxel_idx] = max / frequencies[max_idx];
  }
}

__global__ void setTexVoxelDirs(const TexVoxel::Dir* dirs,
                                const float* smooth_weights,
                                TexBlock** tex_block_ptrs) {
  // Get the Voxel we'll check in this thread
  TexBlock* tex_block = tex_block_ptrs[blockIdx.x];
  Index3D voxel_index = Index3D(threadIdx.z, threadIdx.y, threadIdx.x);
  TexVoxel* tex_voxel =
      &(tex_block->voxels[voxel_index[0]][voxel_index[1]][voxel_index[2]]);

  const int linear_idx = linearizedThreadVoxelIdx<TexBlock>();
  const TexVoxel::Dir smooth_dir = dirs[linear_idx];
  if (tex_voxel->dir != smooth_dir) {
    tex_voxel->updateDir(smooth_dir, smooth_weights[linear_idx]);
  }
}

void updateTexVoxelDirectionsGPU(
    const GPULayerView<TsdfBlock>& tsdf_layer_view,
    const GPULayerView<TexBlock>& tex_layer_view,
    device_vector<TexBlock*>& tex_block_ptrs,
    const device_vector<Index3D>& block_indices_device, const int num_blocks,
    const cudaStream_t stream, const float block_size, const float voxel_size) {
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_blocks;

  // Update TexVoxel directions in two steps:
  // 1. Update TexVoxel directions based on the TSDF surface gradient
  // 2. Comptue the smoothed directions with majority voting
  // 3. Update all TexVoxels to their smoothed directions
  //
  // We do this in two separate calls to guarantee that all TexVoxel directions
  // are set before smoothing.

  // clang-format off
  setTexVoxelDirsfromTsdfGradient<<<num_thread_blocks, kThreadsPerBlock, 0, stream>>>(
      tsdf_layer_view.getHash().impl_,
      tex_block_ptrs.data(),
      block_indices_device.data(),
      block_size,
      voxel_size
  );
  // clang-format on
  checkCudaErrors(cudaStreamSynchronize(stream));
  checkCudaErrors(cudaPeekAtLastError());

  constexpr int voxels_per_block = TexBlock::kVoxelsPerSide *
                                   TexBlock::kVoxelsPerSide *
                                   TexBlock::kVoxelsPerSide;
  device_vector<TexVoxel::Dir> smooth_dirs;
  smooth_dirs.reserve(block_indices_device.size() * voxels_per_block);
  device_vector<float> smooth_weights;
  smooth_weights.reserve(block_indices_device.size() * voxels_per_block);

  // clang-format off
  majorityVoteTexVoxelDirs<<<num_thread_blocks, kThreadsPerBlock, 0, stream>>>(
      tex_layer_view.getHash().impl_,
      tex_block_ptrs.data(),
      block_indices_device.data(),
      block_size,
      voxel_size,
      smooth_dirs.data(),
      smooth_weights.data()
  );
  // clang-format on
  checkCudaErrors(cudaStreamSynchronize(stream));
  checkCudaErrors(cudaPeekAtLastError());

  // clang-format off
  setTexVoxelDirs<<<num_thread_blocks, kThreadsPerBlock, 0, stream>>>(
    smooth_dirs.data(), 
    smooth_weights.data(),
    tex_block_ptrs.data()
  );
  // clang-format on
  checkCudaErrors(cudaStreamSynchronize(stream));
  checkCudaErrors(cudaPeekAtLastError());
}

}  // namespace tex
}  // namespace nvblox
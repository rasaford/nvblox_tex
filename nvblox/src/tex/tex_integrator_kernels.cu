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
__device__ inline bool getTsdfVoxelValue(
    const Index3DDeviceHashMapType<TsdfBlock>& gpu_hash_index_to_str,
    const Vector3f& position, const float block_size, float& distance) {
  TsdfVoxel* voxel;
  if (getVoxelAtPosition<TsdfVoxel>(gpu_hash_index_to_str, position, block_size,
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
  valid &= getTsdfVoxelValue(gpu_hash_index_to_ptr, position + Vector3f(voxel_size, 0.0f, 0.0f), block_size, dist_x_plus); 
  valid &= getTsdfVoxelValue(gpu_hash_index_to_ptr, position + Vector3f(0.0f, voxel_size, 0.0f), block_size, dist_y_plus); 
  valid &= getTsdfVoxelValue(gpu_hash_index_to_ptr, position + Vector3f(0.0f, 0.0f, voxel_size), block_size, dist_z_plus); 
  valid &= getTsdfVoxelValue(gpu_hash_index_to_ptr, position - Vector3f(voxel_size, 0.0f, 0.0f), block_size, dist_x_minus); 
  valid &= getTsdfVoxelValue(gpu_hash_index_to_ptr, position - Vector3f(0.0f, voxel_size, 0.0f), block_size, dist_y_minus); 
  valid &= getTsdfVoxelValue(gpu_hash_index_to_ptr, position - Vector3f(0.0f, 0.0f, voxel_size), block_size, dist_z_minus);
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
 * @brief Updates the direction values for all TexVoxels where this has not
 * been set yet
 *
 * @param tsdf_block_ptrs
 * @param tex_block_ptrs
 * @param block_size
 */
__global__ void setTexVoxelDirections(
    const Index3DDeviceHashMapType<TsdfBlock> gpu_hash_index_to_ptr,
    TexBlock** tex_block_ptrs, const Index3D* block_indices,
    const float block_size, const float voxel_size) {
  // Get the Voxels we'll check in this thread
  TexBlock* tex_block = tex_block_ptrs[blockIdx.x];
  const Index3D& block_index = block_indices[blockIdx.x];
  Index3D voxel_index = Index3D(threadIdx.z, threadIdx.y, threadIdx.x);
  TexVoxel* tex_voxel =
      &(tex_block->voxels[voxel_index[0]][voxel_index[1]][voxel_index[2]]);

  // only update the direction of newly allocated voxels
  if (tex_voxel->isInitialized()) return;

  Vector3f position = getCenterPostionFromBlockIndexAndVoxelIndex(
      block_size, block_index, voxel_index);

  // Since we are working in an TSDF, where the distance of each voxel to the
  // surface implicitly defines the surface boundary, the normal of each
  //   voxel
  // is just the normalized gradient.
  Vector3f gradient;
  const bool valid = computeTSDFGradient(gpu_hash_index_to_ptr, position,
                                         block_size, voxel_size, gradient);
  const double gradient_norm = gradient.norm();
  if (valid && gradient_norm > 0) {
    tex_voxel->dir = quantizeDirection(gradient / gradient_norm);
    // printf("gradient: (%f %f %f), dir: %d\n", gradient[0], gradient[1],
    // gradient[2], tex_voxel.dir);
  }
}

void updateTexVoxelDirectionsGPU(
    const GPULayerView<TsdfBlock> gpu_layer,
    device_vector<TexBlock*>& tex_block_ptrs,
    const device_vector<Index3D>& block_indices_device, const int num_blocks,
    const cudaStream_t stream, const float block_size, const float voxel_size) {
  // Do the check on GPU
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_blocks;
  // clang-format off
  setTexVoxelDirections<<<num_thread_blocks, kThreadsPerBlock, 0, stream>>>(
      gpu_layer.getHash().impl_,
      tex_block_ptrs.data(),
      block_indices_device.data(),
      block_size,
      voxel_size);
  // clang-format on
  checkCudaErrors(cudaStreamSynchronize(stream));
  checkCudaErrors(cudaPeekAtLastError());
}

}  // namespace tex
}  // namespace nvblox
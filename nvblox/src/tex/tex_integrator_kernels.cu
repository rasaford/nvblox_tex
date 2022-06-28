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
 * @brief Gets the voxel at the given index from a potentially neighboring
 * block. E.g. the given voxel_index can be out of range of the current block.
 * So it is possible that voxel_index < 0 or voxel_index >
 * TsdfBlock::kVoxelsPerSide at any coordinate.
 *
 * However, this function is only defined if the given voxel_index is in the
 * current block (given by blockIdx.x) or one of the 27 directly neighboring
 * ones. If it is further away a nullptr is returned.
 *
 * @param blocks 2D array of neighbor block pointers
 * @param voxel_index voxel_index to get the voxel at
 * @return const TsdfVoxel* if the voxel is at most one block away from
 * the current block and has been allocated, else nullptr.
 */
__device__ const TsdfVoxel* getVoxel(const TsdfBlock** blocks,
                                     const Index3D& voxel_index) {
  const int block_idx = blockIdx.x;

  // create a local copy of the voxel index, since we are going to be modifying
  // it.
  Index3D voxel_idx = voxel_index;
  Index3D block_offset{0, 0, 0};
  constexpr int voxels_per_side = static_cast<int>(TsdfBlock::kVoxelsPerSide);
  for (int i = 0; i < 3; ++i) {
    if (voxel_idx[i] >= voxels_per_side) {
      voxel_idx[i] -= voxels_per_side;
      block_offset[i] = 1;
    } else if (voxel_idx[i] < 0) {
      voxel_idx[i] += voxels_per_side;
      block_offset[i] = -1;
    }
  }

  // We cannot look more than one block in any direction
  if ((block_offset.array() > 1).any() || (block_offset.array() < -1).any() ||
      (voxel_idx.array() < 0).any() ||
      (voxel_idx.array() >= TsdfBlock::kVoxelsPerSide).any()) {
    return nullptr;
  }
  // get the neighboring voxel either from a neighboring block if it's outside
  // the current one, or get it directly from the current block
  int linear_neighbor_idx =
      neighbor::neighborBlockIndexFromOffset(block_offset);
  const TsdfBlock* block =
      blocks[block_idx * neighbor::kCubeNeighbors + linear_neighbor_idx];

  if (block == nullptr) {
    return nullptr;
  }
  return &block->voxels[voxel_idx.x()][voxel_idx.y()][voxel_idx.z()];
}

/**
 * @brief Performs trilinear interpolation of the surface values at the given
 * position.
 *
 * @param blocks neighbor block pointers
 * @param position has to be inside the voxel defined by voxel_index
 * @param voxel_index voxel index of position
 * @param dist output interpolate surface distance
 * @return __device__ if the output is valid
 */
__device__ bool trilinearInterpolation(const TsdfBlock** blocks,
                                       const Vector3f& position,
                                       const Index3D voxel_index,
                                       const float voxel_size, float* dist) {
  const Vector3f normalized_pos = position / voxel_size;
  const Vector3f weight =
      normalized_pos.array() - normalized_pos.array().floor();

  (*dist) = 0.f;
  const TsdfVoxel* v;
  // clang-format off
  v = getVoxel(blocks, voxel_index + Index3D(0, 0, 0));   if (v == nullptr || v->weight == 0.f) return false;   (*dist) += (1.f - weight.x()) * (1.f - weight.y()) * (1.f - weight.z()) * v->distance;
  v = getVoxel(blocks, voxel_index + Index3D(1, 0, 0));   if (v == nullptr || v->weight == 0.f) return false;   (*dist) += weight.x()         * (1.f - weight.y()) * (1.f - weight.z()) * v->distance;
  v = getVoxel(blocks, voxel_index + Index3D(0, 1, 0));   if (v == nullptr || v->weight == 0.f) return false;   (*dist) += (1.f - weight.x()) * weight.y()         * (1.f - weight.z()) * v->distance;
  v = getVoxel(blocks, voxel_index + Index3D(0, 0, 1));   if (v == nullptr || v->weight == 0.f) return false;   (*dist) += (1.f - weight.x()) * (1.f - weight.y()) *  weight.z()        * v->distance;
  v = getVoxel(blocks, voxel_index + Index3D(1, 1, 0));   if (v == nullptr || v->weight == 0.f) return false;   (*dist) += weight.x()         * weight.y()         * (1.f - weight.z()) * v->distance;
  v = getVoxel(blocks, voxel_index + Index3D(0, 1, 1));   if (v == nullptr || v->weight == 0.f) return false;   (*dist) += (1.f - weight.x()) * weight.y()         *  weight.z()        * v->distance;
  v = getVoxel(blocks, voxel_index + Index3D(1, 0, 1));   if (v == nullptr || v->weight == 0.f) return false;   (*dist) += weight.x()         * (1.f - weight.y()) *  weight.z()        * v->distance;
  v = getVoxel(blocks, voxel_index + Index3D(1, 1, 1));   if (v == nullptr || v->weight == 0.f) return false;   (*dist) += weight.x()         *  weight.y()        *  weight.z()        * v->distance;
  // clang-format on

  return true;
}

/**
 * @brief Computes TSDF gradient for the current position.
 *
 * @param neighbor_blocks
 * @param voxel_index index of the voxel the position is in
 * @param position position to compute gradient at
 * @param voxel_size
 * @param gradient output gradient, only written to if the return value of this
 * function is true
 * @return __device__ if the computed gradient is valid, e.g. all neighbor
 * blocks for interpolation exist
 */
__device__ bool computeTSDFGradient(const TsdfBlock** neighbor_blocks,
                                    const Index3D& voxel_index,
                                    const Vector3f& position,
                                    const float voxel_size,
                                    Vector3f& gradient) {
  // voxel size is block size divided by number of blocks per side
  // const float voxel_size =
  //     block_size / static_cast<float>(tsdf_block->kVoxelsPerSide);
  const float voxel_size_half = 0.5f * voxel_size;
  const float v_quarter = 0.25f * voxel_size;
  float dist_x_plus = 0, dist_x_minus = 0, dist_y_plus = 0, dist_y_minus = 0,
        dist_z_plus = 0, dist_z_minus = 0;
  bool valid = true;

  const TsdfVoxel* v;
  // get tsdf values for each neighboring voxel to the current one
  // clang-format off
  valid &= trilinearInterpolation(neighbor_blocks, position + Vector3f(v_quarter, 0.f,        0.f), voxel_index, voxel_size, &dist_x_plus);
  valid &= trilinearInterpolation(neighbor_blocks, position + Vector3f(0.f,       v_quarter,  0.f), voxel_index, voxel_size, &dist_y_plus);
  valid &= trilinearInterpolation(neighbor_blocks, position + Vector3f(0.f,       0.f,  v_quarter), voxel_index, voxel_size, &dist_z_plus);
  valid &= trilinearInterpolation(neighbor_blocks, position - Vector3f(v_quarter, 0.f,        0.f), voxel_index, voxel_size, &dist_x_minus);
  valid &= trilinearInterpolation(neighbor_blocks, position - Vector3f(0.f,       v_quarter,  0.f), voxel_index, voxel_size, &dist_y_minus);
  valid &= trilinearInterpolation(neighbor_blocks, position - Vector3f(0.f,       0.f,  v_quarter), voxel_index, voxel_size, &dist_z_minus);
  // clang-format on

  if (!valid) {
    return false;
  }

  // approximate gradient by finite differences
  gradient << (dist_x_plus - dist_x_minus) / voxel_size_half,
      (dist_y_plus - dist_y_minus) / voxel_size_half,
      (dist_z_plus - dist_z_minus) / voxel_size_half;
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
 * @brief Updates the direction values for all given TexVoxels where the
 * confidence of a direction increased enough
 *
 * @param neighbor_blocks
 * @param tex_block_ptrs
 * @param block_indices
 * @param block_size
 * @param voxel_size
 * @return __global__
 */
__global__ void setTexVoxelDirsfromTsdfGradient(
    const TsdfBlock** neighbor_blocks, TexBlock** tex_block_ptrs,
    const Index3D* block_indices, const float block_size,
    const float voxel_size) {
  // Get the Voxels we'll check in this thread
  TexBlock* tex_block = tex_block_ptrs[blockIdx.x];
  Index3D block_idx = block_indices[blockIdx.x];
  Index3D voxel_idx = Index3D(threadIdx.z, threadIdx.y, threadIdx.x);
  TexVoxel* tex_voxel =
      &(tex_block->voxels[voxel_idx[0]][voxel_idx[1]][voxel_idx[2]]);

  // only update the direction for voxels where we are not yet very confident in
  // thier direction
  // if (tex_voxel->dir_weight >= TexVoxel::DIR_THRESHOLD) return;

  Vector3f position = getCenterPostionFromBlockIndexAndVoxelIndex(
      block_size, block_idx, voxel_idx);

  // Since we are working in an TSDF, where the distance of each voxel to the
  // surface implicitly defines the surface boundary, the normal of each voxel
  // is just the normalized gradient.
  Vector3f gradient;
  const bool valid_gradient = computeTSDFGradient(
      neighbor_blocks, voxel_idx, position, voxel_size, gradient);
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
    tex_voxel->updateDir(dir, dir_weight);
  }
}

/**
 * @brief Updates the TexVoxel directions for each block_index.
 *
 * @param neighbor_blocks 2D array of neighboring block pointers, see callsite
 * on how to initialize it
 * @param block_indices indices of the blocks to process
 * @param tex_block_ptrs TexBlocks which will be modified
 * @param num_blocks number of TsdfBocks
 * @param stream cuda integration stream
 * @param block_size world block size
 * @param voxel_size world voxel size
 */
void updateTexVoxelDirectionsGPU(
    device_vector<const TsdfBlock*> neighbor_blocks,
    const device_vector<Index3D> block_indices,
    device_vector<TexBlock*>& tex_block_ptrs, const int num_blocks,
    const cudaStream_t stream, const float block_size, const float voxel_size) {
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_blocks;

  // clang-format off
  setTexVoxelDirsfromTsdfGradient<<<num_thread_blocks, kThreadsPerBlock, 0, stream>>>(
      neighbor_blocks.data(),
      tex_block_ptrs.data(),
      block_indices.data(),
      block_size,
      voxel_size
  );
  // clang-format on
  checkCudaErrors(cudaStreamSynchronize(stream));
  checkCudaErrors(cudaPeekAtLastError());
}

}  // namespace tex
}  // namespace nvblox
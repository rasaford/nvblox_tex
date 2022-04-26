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
#include <helper_math.h>
#include "nvblox/integrators/projective_tex_integrator.h"

#include "nvblox/integrators/cuda/projective_integrators_common.cuh"
#include "nvblox/integrators/integrators_common.h"
#include "nvblox/utils/timing.h"
namespace nvblox {

ProjectiveTexIntegrator::ProjectiveTexIntegrator()
    : ProjectiveIntegratorBase() {
  sphere_tracer_.params().maximum_ray_length_m = max_integration_distance_m_;
  checkCudaErrors(cudaStreamCreate(&integration_stream_));
}

ProjectiveTexIntegrator::~ProjectiveTexIntegrator() {
  finish();
  checkCudaErrors(cudaStreamDestroy(integration_stream_));
}

void ProjectiveTexIntegrator::finish() const {
  cudaStreamSynchronize(integration_stream_);
}

void ProjectiveTexIntegrator::integrateFrame(
    const ColorImage& color_frame, const Transform& T_L_C, const Camera& camera,
    const TsdfLayer& tsdf_layer, TexLayer* tex_layer,
    std::vector<Index3D>* updated_blocks) {
  CHECK_NOTNULL(tex_layer);
  CHECK_EQ(tsdf_layer.block_size(), tex_layer->block_size());

  // Metric truncation distance for this layer
  const float voxel_size =
      tex_layer->block_size() / VoxelBlock<bool>::kVoxelsPerSide;
  const float truncation_distance_m = truncation_distance_vox_ * voxel_size;

  timing::Timer blocks_in_view_timer("tex/integrate/get_blocks_in_view");
  std::vector<Index3D> block_indices =
      getBlocksInView(T_L_C, camera, tex_layer->block_size());
  blocks_in_view_timer.Stop();

  // Check which of these blocks are:
  // - Allocated in the TSDF, and
  // - have at least a single voxel within the truncation band
  // This is because:
  // - We don't allocate new geometry here, we just color existing geometry
  // - We don't color freespace.
  timing::Timer blocks_in_band_timer("tex/integrate/reduce_to_blocks_in_band");
  block_indices = reduceBlocksToThoseInTruncationBand(block_indices, tsdf_layer,
                                                      truncation_distance_m);
  blocks_in_band_timer.Stop();

  // Allocate blocks (CPU)
  // We allocate color blocks where
  // - there are allocated TSDF blocks, AND
  // - these blocks are within the truncation band
  timing::Timer allocate_blocks_timer("tex/integrate/allocate_blocks");
  allocateBlocksWhereRequired(block_indices, tex_layer);
  allocate_blocks_timer.Stop();

  // Update normal directions for all voxels which do not have a voxel dir set
  // already
  timing::Timer update_normals_timer("tex/integrate/update_normals");
  updateVoxelNormalDirections(tsdf_layer, tex_layer, block_indices,
                              truncation_distance_m);
  update_normals_timer.Stop();

  // Create a synthetic depth image
  timing::Timer sphere_trace_timer("tex/integrate/sphere_trace");
  std::shared_ptr<const DepthImage> synthetic_depth_image_ptr =
      sphere_tracer_.renderImageOnGPU(
          camera, T_L_C, tsdf_layer, truncation_distance_m, MemoryType::kDevice,
          depth_render_ray_subsampling_factor_);
  sphere_trace_timer.Stop();

  // Update identified blocks
  // Calls out to the child-class implementing the integation (GPU)
  timing::Timer update_blocks_timer("tex/integrate/update_blocks");
  updateBlocks(block_indices, color_frame, *synthetic_depth_image_ptr, T_L_C,
               camera, truncation_distance_m, tex_layer);
  update_blocks_timer.Stop();

  if (updated_blocks != nullptr) {
    *updated_blocks = block_indices;
  }
}

// __device__ inline Color blendTwoColors(const Color& first_color,
//                                        float first_weight,
//                                        const Color& second_color,
//                                        float second_weight) {
//   float total_weight = first_weight + second_weight;

//   first_weight /= total_weight;
//   second_weight /= total_weight;

//   Color new_color;
//   new_color.r = static_cast<uint8_t>(std::round(
//       first_color.r * first_weight + second_color.r * second_weight));
//   new_color.g = static_cast<uint8_t>(std::round(
//       first_color.g * first_weight + second_color.g * second_weight));
//   new_color.b = static_cast<uint8_t>(std::round(
//       first_color.b * first_weight + second_color.b * second_weight));

//   return new_color;
// }

// __device__ inline bool updateVoxel(const Color color_measured,
//                                    ColorVoxel* voxel_ptr,
//                                    const float voxel_depth_m,
//                                    const float truncation_distance_m,
//                                    const float max_weight) {
//   // NOTE(alexmillane): We integrate all voxels passed to this function, We
//   // should probably not do this. We should no update some based on occlusion
//   // and their distance in the distance field....
//   // TODO(alexmillane): The above.

//   // Read CURRENT voxel values (from global GPU memory)
//   const Color voxel_color_current = voxel_ptr->color;
//   const float voxel_weight_current = voxel_ptr->weight;
//   // Fuse
//   constexpr float measurement_weight = 1.0f;
//   const Color fused_color =
//       blendTwoColors(voxel_color_current, voxel_weight_current, color_measured,
//                      measurement_weight);
//   const float weight =
//       fmin(measurement_weight + voxel_weight_current, max_weight);
//   // Write NEW voxel values (to global GPU memory)
//   voxel_ptr->color = fused_color;
//   voxel_ptr->weight = weight;
//   return true;
// }

__global__ void integrateBlocks(
    const Index3D* block_indices_device_ptr, const Camera camera,
    const Color* color_image, const int color_rows, const int color_cols,
    const float* depth_image, const int depth_rows, const int depth_cols,
    const Transform T_C_L, const float block_size,
    const float truncation_distance_m, const float max_weight,
    const float max_integration_distance, const int depth_subsample_factor,
    TexBlock** block_device_ptrs) {
  // TODO (rasaford)
  // - Here we need to determine the TexVoxel direction first. (one projection
  // is enough)
  // - Backproject every pixel in the TexVoxel to the image and update it's
  // color

  // Get - the image-space projection of the voxel center associated with this
  // thread
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  if (!projectThreadVoxel(block_indices_device_ptr, camera, T_C_L, block_size,
                          &u_px, &voxel_depth_m)) {
    return;
  }

  // If voxel further away than the limit, skip this voxel
  if (max_integration_distance > 0.0f) {
    if (voxel_depth_m > max_integration_distance) {
      return;
    }
  }

  // Get - the depth of the voxel center
  //     - Also check if the voxel projects inside the image
  const Eigen::Vector2f u_px_depth =
      u_px / static_cast<float>(depth_subsample_factor);
  float surface_depth_m;
  if (!interpolation::interpolate2DLinear<float>(
          depth_image, u_px_depth, depth_rows, depth_cols, &surface_depth_m)) {
    return;
  }

  // Occlusion testing
  // Get the distance of the voxel from the rendered surface. If outside
  // truncation band, skip.
  const float voxel_distance_from_surface = surface_depth_m - voxel_depth_m;
  if (fabsf(voxel_distance_from_surface) > truncation_distance_m) {
    return;
  }

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order such
  // that adjacent threads (x-major) access adjacent memory locations in the
  // block (z-major).
  TexVoxel* voxel_ptr = &(block_device_ptrs[blockIdx.x]
                              ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // loop over all colors in the TexVoxel patch
  for (int row = 0; row < voxel_ptr->kPatchWidth; ++row) {
    for (int col = 0; col < voxel_ptr->kPatchWidth; ++col) {
      // u_px =
    }
  }
  Color image_value;
  if (!interpolation::interpolate2DLinear<Color>(color_image, u_px, color_rows,
                                                 color_cols, &image_value)) {
    return;
  }

  // Update the voxel using the update rule for this layer type
  // updateVoxel(image_value, voxel_ptr, voxel_depth_m, truncation_distance_m,
  //             max_weight);
}

void ProjectiveTexIntegrator::updateBlocks(
    const std::vector<Index3D>& block_indices, const ColorImage& color_frame,
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    const float truncation_distance_m, TexLayer* layer_ptr) {
  CHECK_NOTNULL(layer_ptr);
  CHECK_EQ(color_frame.rows() % depth_frame.rows(), 0);
  CHECK_EQ(color_frame.cols() % depth_frame.cols(), 0);

  if (block_indices.empty()) {
    return;
  }
  const int num_blocks = block_indices.size();
  const int depth_subsampling_factor = color_frame.rows() / depth_frame.rows();
  CHECK_EQ(color_frame.cols() / depth_frame.cols(), depth_subsampling_factor);

  // Expand the buffers when needed
  if (num_blocks > block_indices_device_.size()) {
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks);
    block_indices_device_.reserve(new_size);
    block_ptrs_device_.reserve(new_size);
    block_indices_host_.reserve(new_size);
    block_ptrs_host_.reserve(new_size);
  }

  // Stage on the host pinned memory
  block_indices_host_ = block_indices;
  block_ptrs_host_ = getBlockPtrsFromIndices(block_indices, layer_ptr);

  // Transfer to the device
  block_indices_device_ = block_indices_host_;
  block_ptrs_device_ = block_ptrs_host_;

  // We need the inverse transform in the kernel
  const Transform T_C_L = T_L_C.inverse();

  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = block_indices.size();
  // clang-format off
  integrateBlocks<<<num_thread_blocks, kThreadsPerBlock, 0, integration_stream_>>>(
      block_indices_device_.data(),
      camera,
      color_frame.dataConstPtr(),
      color_frame.rows(),
      color_frame.cols(),
      depth_frame.dataConstPtr(),
      depth_frame.rows(),
      depth_frame.cols(),
      T_C_L,
      layer_ptr->block_size(),
      truncation_distance_m,
      max_weight_,
      max_integration_distance_m_,
      depth_subsampling_factor,
      block_ptrs_device_.data());
  // clang-format on
  checkCudaErrors(cudaPeekAtLastError());

  // Finish processing of the frame before returning control
  finish();
}

__global__ void checkBlocksInTruncationBandTex(
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

std::vector<Index3D>
ProjectiveTexIntegrator::reduceBlocksToThoseInTruncationBand(
    const std::vector<Index3D>& block_indices, const TsdfLayer& tsdf_layer,
    const float truncation_distance_m) {
  // Check 1) Are the blocks allocated
  // - performed on the CPU because the hash-map is on the CPU
  std::vector<Index3D> block_indices_check_1;
  block_indices_check_1.reserve(block_indices.size());
  for (const Index3D& block_idx : block_indices) {
    if (tsdf_layer.isBlockAllocated(block_idx)) {
      block_indices_check_1.push_back(block_idx);
    }
  }

  if (block_indices_check_1.empty()) {
    return block_indices_check_1;
  }

  // Check 2) Does each of the blocks have a voxel within the truncation band
  // - performed on the GPU because the blocks are there
  // Get the blocks we need to check
  std::vector<const TsdfBlock*> block_ptrs =
      getBlockPtrsFromIndices(block_indices_check_1, tsdf_layer);

  const int num_blocks = block_ptrs.size();

  // Expand the buffers when needed
  if (num_blocks > truncation_band_block_ptrs_device_.size()) {
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks);
    truncation_band_block_ptrs_host_.reserve(new_size);
    truncation_band_block_ptrs_device_.reserve(new_size);
    block_in_truncation_band_device_.reserve(new_size);
    block_in_truncation_band_host_.reserve(new_size);
  }

  // Host -> Device
  truncation_band_block_ptrs_host_ = block_ptrs;
  truncation_band_block_ptrs_device_ = truncation_band_block_ptrs_host_;

  // Prepare output space
  block_in_truncation_band_device_.resize(num_blocks);

  // Do the check on GPU
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_blocks;
  // clang-format off
  checkBlocksInTruncationBandTex<<<num_thread_blocks, kThreadsPerBlock, 0, integration_stream_>>>(
      truncation_band_block_ptrs_device_.data(),
      truncation_distance_m,
      block_in_truncation_band_device_.data());
  // clang-format on
  checkCudaErrors(cudaStreamSynchronize(integration_stream_));
  checkCudaErrors(cudaPeekAtLastError());

  // Copy results back
  block_in_truncation_band_host_ = block_in_truncation_band_device_;

  // Filter the indices using the result
  std::vector<Index3D> block_indices_check_2;
  block_indices_check_2.reserve(block_indices_check_1.size());
  for (int i = 0; i < block_indices_check_1.size(); i++) {
    if (block_in_truncation_band_host_[i]) {
      block_indices_check_2.push_back(block_indices_check_1[i]);
    }
  }

  return block_indices_check_2;
}

template <typename BlockType>
__device__ inline bool isValidBlockIndex(const int x, const int y,
                                         const int z) {
  // Check if the given voxel index is within the current block
  if (x < 0 || x >= BlockType::kVoxelsPerSide) {
    return false;
  }
  if (y < 0 || y >= BlockType::kVoxelsPerSide) {
    return false;
  }
  if (z < 0 || z >= BlockType::kVoxelsPerSide) {
    return false;
  }
  return true;
}

/**
 * @brief Get the TSDF value at the current voxel, if the given voxel index is
 * still within the given block
 *
 * @param tsdf_block
 * @param x
 * @param y
 * @param z
 * @param distance output distance. Only written to if the return value == true
 * @return if the voxel index is within the given block
 */
__device__ inline bool getTsdfVoxelValue(const TsdfBlock* tsdf_block,
                                         const int x, const int y, const int z,
                                         float& distance) {
  if (!isValidBlockIndex<TsdfBlock>(x, y, z)) return;

  const TsdfVoxel& voxel = tsdf_block->voxels[x][y][z];
  // TODO(rasaford) take voxel weight into account here
  // distance = voxel.distance * voxel.weight;
  distance = voxel.distance;
  return true;
}

/**
 * @brief Computes TSDF gradient for the current vertex given by threadIdx.
 * If a vertex does not have 6 neighbors, the gradient will be (0, 0, 0).
 *
 * @param tsdf_block
 * @param block_size
 * @param gradient 
 */
__device__ bool computeTSDFGradient(const TsdfBlock* tsdf_block,
                                    const float block_size,
                                    Vector3f& gradient) {
  // voxel size is block size divided by number of blocks per side
  const float voxel_size =
      block_size / static_cast<float>(tsdf_block->kVoxelsPerSide);
  const float voxel_size_2x = 2 * voxel_size;
  // the voxel index is the current threadIndex
  const int3 voxel_idx = make_int3(threadIdx);
  float dist_x_plus = 0, dist_x_minus = 0, dist_y_plus = 0, dist_y_minus = 0,
        dist_z_plus = 0, dist_z_minus = 0;
  bool valid = true;
  // get tsdf values for each neighboring voxel to the current one within the
  // given block
  // clang-format off
  valid &= getTsdfVoxelValue(tsdf_block, voxel_idx.x + 1, voxel_idx.y,      voxel_idx.z,      dist_x_plus);
  valid &= getTsdfVoxelValue(tsdf_block, voxel_idx.x,     voxel_idx.y + 1,  voxel_idx.z,      dist_y_plus);
  valid &= getTsdfVoxelValue(tsdf_block, voxel_idx.x,     voxel_idx.y,      voxel_idx.z + 1,  dist_z_plus);
  valid &= getTsdfVoxelValue(tsdf_block, voxel_idx.x - 1, voxel_idx.y,      voxel_idx.z,      dist_x_minus);
  valid &= getTsdfVoxelValue(tsdf_block, voxel_idx.x,     voxel_idx.y - 1,  voxel_idx.z,      dist_y_minus);
  valid &= getTsdfVoxelValue(tsdf_block, voxel_idx.x,     voxel_idx.y,      voxel_idx.z - 1,  dist_z_minus);
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
 * TexVoxelDir directions
 *
 * @param normal **normalized** direcction vector
 * @return quantized direction
 */
__device__ inline TexVoxelDir quantizeDirection(const Vector3f& dir) {
  const Vector3f abs_dir = dir.cwiseAbs();
  TexVoxelDir res;
  if (abs_dir(0) >= abs_dir(1) && abs_dir(0) >= abs_dir(2)) {
    res = dir(0) < 0 ? TexVoxelDir::X_MINUS : res = TexVoxelDir::X_PLUS;
  } else if (abs_dir(1) >= abs_dir(2)) {
    res = dir(1) < 0 ? TexVoxelDir::Y_MINUS : TexVoxelDir::Y_PLUS;
  } else {
    res = dir(2) < 0 ? TexVoxelDir::Z_MINUS : TexVoxelDir::Z_PLUS;
  }
  return res;
}

/**
 * @brief Updates the direction values for all TexVoxels where this has not been
 * set yet
 *
 * @param tsdf_block_ptrs
 * @param tex_block_ptrs
 * @param block_size
 */
__global__ void setTexVoxelDirections(
    const VoxelBlock<TsdfVoxel>** tsdf_block_ptrs,
    const VoxelBlock<TexVoxel>** tex_block_ptrs, const float block_size) {
  // Get the Voxels we'll check in this thread
  const TexBlock* tex_block = tex_block_ptrs[blockIdx.x];
  TexVoxel tex_voxel = tex_block->voxels[threadIdx.z][threadIdx.y][threadIdx.x];
  // only update the direction of newly allocated voxels
  if (tex_voxel.dir != TexVoxelDir::NONE) return;

  const TsdfBlock* tsdf_block = tsdf_block_ptrs[blockIdx.x];
  const TsdfVoxel tsdf_voxel =
      tsdf_block->voxels[threadIdx.z][threadIdx.y][threadIdx.x];

  // Since we are working in an TSDF, where the distance of each voxel to the
  // surface implicitly defines the surface boundary, the normal of each voxel
  // is just the normalized gradient.
  Vector3f gradient;
  const bool valid  = computeTSDFGradient(tsdf_block, block_size, gradient);
  if (valid) {
    tex_voxel.dir =  quantizeDirection(gradient.normalized());
  }

  // TODO: (rasaford) this is still quite hacky. Remove the Block abstraction
  // for TexVoxels to limit the number of voxels we have to do this for for

  // Voxels on the edge of a block, determine the direction by majority voting
  // of the neighboring blocks. We sync all threads processing a block here,
  // such that all other direction values have already been written.
  __syncthreads();
  if (tex_voxel.dir == TexVoxelDir::NONE) {
    const int3 voxel_idx = make_int3(threadIdx);
    int frequencies[TexVoxelDir_count] = {};
    // clang-format off
    if(isValidBlockIndex<TsdfBlock>(voxel_idx.x + 1, voxel_idx.y, voxel_idx.z)) {
      frequencies[static_cast<int>(tex_block->voxels[voxel_idx.x + 1][voxel_idx.y][voxel_idx.z].dir)]++;
    }
    if(isValidBlockIndex<TsdfBlock>(voxel_idx.x, voxel_idx.y + 1, voxel_idx.z)) {
      frequencies[static_cast<int>(tex_block->voxels[voxel_idx.x][voxel_idx.y + 1][voxel_idx.z].dir)]++;
    }
    if(isValidBlockIndex<TsdfBlock>(voxel_idx.x, voxel_idx.y, voxel_idx.z + 1)) {
      frequencies[static_cast<int>(tex_block->voxels[voxel_idx.x][voxel_idx.y][voxel_idx.z + 1].dir)]++;
    }
    if(isValidBlockIndex<TsdfBlock>(voxel_idx.x - 1, voxel_idx.y, voxel_idx.z)) {
      frequencies[static_cast<int>(tex_block->voxels[voxel_idx.x - 1][voxel_idx.y][voxel_idx.z].dir)]++;
    }
    if(isValidBlockIndex<TsdfBlock>(voxel_idx.x, voxel_idx.y - 1, voxel_idx.z)) {
      frequencies[static_cast<int>(tex_block->voxels[voxel_idx.x][voxel_idx.y - 1][voxel_idx.z].dir)]++;
    }
    if(isValidBlockIndex<TsdfBlock>(voxel_idx.x, voxel_idx.y, voxel_idx.z - 1)) {
      frequencies[static_cast<int>(tex_block->voxels[voxel_idx.x][voxel_idx.y][voxel_idx.z - 1].dir)]++;
    }
    // clang-format on
    int max_freq_idx = 0;
    int max_freq = 0;
    for (int i = 0; i < TexVoxelDir_count; ++i) {
      if (frequencies[i] > max_freq) {
        max_freq = frequencies[i];
        max_freq_idx = i;
      }
    }
    tex_voxel.dir = static_cast<TexVoxelDir>(max_freq_idx);
  }
}

void ProjectiveTexIntegrator::updateVoxelNormalDirections(
    const TsdfLayer& tsdf_layer, const TexLayer* tex_layer_ptr,
    const std::vector<Index3D>& block_indices,
    const float truncation_distance_m) {
  const int num_blocks = block_indices.size();

  // Get the pointers for the indexed blocks from both
  // - The tsdf layer: Since all Voxels are already integrated here, we read
  // from this layer to estimate the normal direcitonA
  // - The TexBlock latyer: We write the updated directions to this layer for
  // all new blocks
  std::vector<const TsdfBlock*> tsdf_block_ptrs =
      getBlockPtrsFromIndices(block_indices, tsdf_layer);
  std::vector<const TexBlock*> tex_block_ptrs =
      getBlockPtrsFromIndices(block_indices, *tex_layer_ptr);

  // Expand the buffers when needed
  if (num_blocks > update_normals_tex_block_prts_device_.size()) {
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks);
    update_normals_tex_block_prts_device_.reserve(new_size);
    update_normals_tex_block_prts_host_.reserve(new_size);
    update_normals_tsdf_block_prts_device_.reserve(new_size);
    update_normals_tsdf_block_prts_host_.reserve(new_size);
  }

  // Host -> Device
  update_normals_tex_block_prts_host_ = tex_block_ptrs;
  update_normals_tex_block_prts_device_ = update_normals_tex_block_prts_host_;
  update_normals_tsdf_block_prts_host_ = tsdf_block_ptrs;
  update_normals_tsdf_block_prts_device_ = update_normals_tsdf_block_prts_host_;

  // Do the check on GPU
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_blocks;
  // clang-format off
  setTexVoxelDirections<<<num_thread_blocks, kThreadsPerBlock, 0, integration_stream_>>>(
      update_normals_tsdf_block_prts_device_.data(),
      update_normals_tex_block_prts_device_.data(), 
      tsdf_layer.block_size());
  // clang-format on
  checkCudaErrors(cudaStreamSynchronize(integration_stream_));
  checkCudaErrors(cudaPeekAtLastError());
}

}  // namespace nvblox

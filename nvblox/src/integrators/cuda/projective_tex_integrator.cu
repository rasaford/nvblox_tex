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
#include "nvblox/integrators/cuda/projective_integrators_common.cuh"
#include "nvblox/integrators/integrators_common.h"
#include "nvblox/integrators/projective_tex_integrator.h"
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

__device__ inline void updateTexel(const Color& color_measured,
                                   TexVoxel* tex_voxel,
                                   const Index2D& texel_idx,
                                   const float voxel_depth_m,
                                   const float truncation_distance_m,
                                   const float max_weight) {
  // NOTE(alexmillane): We integrate all voxels passed to this function, We
  // should probably not do this. We should no update some based on occlusion
  // and their distance in the distance field....
  // TODO(alexmillane): The above.

  // Read CURRENT voxel values (from global GPU memory)
  const Color& texel_color_current = (*tex_voxel)(texel_idx);
  const float texel_weight_current = tex_voxel->weight;
  // TODO: (rasaford) compute measurement weight based on e.g.
  // - size of the projected texel in the image
  // - sharpness of the projected area in the image (to compensate motion blur)
  // - how flat on we're looking at the texel projection
  // - if the texel is on a boundary
  // - ...
  constexpr float measurement_weight = 1.0f;
  const Color fused_color =
      blendTwoColors(texel_color_current, texel_weight_current, color_measured,
                     measurement_weight);
  const float weight =
      fmin(measurement_weight + texel_weight_current, max_weight);
  // Write NEW voxel values (to global GPU memory)
  (*tex_voxel)(texel_idx) = fused_color;
  tex_voxel->weight = weight;
}

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

  // NOTE(rasaford): If the current voxel has not been assigned a texture plane
  // direction, it must not be on the truncation band --> skip it
  if (!voxel_ptr->isInitialized()) {
    return;
  }

  Color image_value;
  // loop over all colors in the TexVoxel patch
  for (int row = 0; row < voxel_ptr->kPatchWidth; ++row) {
    for (int col = 0; col < voxel_ptr->kPatchWidth; ++col) {
      const Index2D texel_idx(row, col);
      // Project the current texel_idx to image space. If it's outside the
      // image, go to the next texel.
      if (!projectThreadTexel(block_indices_device_ptr, camera, T_C_L,
                              block_size, texel_idx, voxel_ptr->dir, &u_px,
                              &voxel_depth_m)) {
        continue;
      }
      // printf("updating color for block (%d, %d, %d)", threadIdx.z,
      // threadIdx.y,
      //        threadIdx.x);
      // get image color at current u_px by linear interpolation
      if (!interpolation::interpolate2DLinear<Color>(
              color_image, u_px, color_rows, color_cols, &image_value)) {
        continue;
      }
      // Update the texel using the update rule for this layer type
      updateTexel(image_value, voxel_ptr, texel_idx, voxel_depth_m,
                  truncation_distance_m, max_weight);
    }
  }
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
  checkBlocksInTruncationBand<<<num_thread_blocks, kThreadsPerBlock, 0, integration_stream_>>>(
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

void ProjectiveTexIntegrator::updateVoxelNormalDirections(
    const TsdfLayer& tsdf_layer, TexLayer* tex_layer_ptr,
    const std::vector<Index3D>& block_indices,
    const float truncation_distance_m) {
  // Get the pointers for the indexed blocks from both
  // - The tsdf layer: Since all Voxels are already integrated here, we read
  // from this layer to estimate the normal direcitonA
  // - The TexBlock layer: We write the updated directions to this layer for
  // all new blocks
  // NOTE(rasaford) Even though we do not modify TsdfLayer, we have to drop
  // const here due to the way the GPUHashMap works
  TsdfLayer* tsdf_layer_non_const = const_cast<TsdfLayer*>(&tsdf_layer);
  std::vector<TsdfBlock*> tsdf_block_ptrs =
      getBlockPtrsFromIndices(block_indices, tsdf_layer_non_const);
  std::vector<TexBlock*> tex_block_ptrs =
      getBlockPtrsFromIndices(block_indices, tex_layer_ptr);

  // We assume that a TsdfBlock at index i corresponds to a TexBlock at i. This
  // cannot be the case if the two vectors don't have the same number of
  // elements
  CHECK_EQ(tsdf_block_ptrs.size(), tex_block_ptrs.size());

  const int num_blocks = block_indices.size();
  // Expand the buffers when needed
  if (num_blocks > update_normals_tex_block_prts_device_.size()) {
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks);
    update_normals_tex_block_prts_device_.reserve(new_size);
    update_normals_tex_block_prts_host_.reserve(new_size);
    update_normals_block_indices_device_.reserve(new_size);
    update_normals_block_indices_host_.reserve(new_size);
  }

  // Host -> Device
  update_normals_tex_block_prts_host_ = tex_block_ptrs;
  update_normals_tex_block_prts_device_ = update_normals_tex_block_prts_host_;
  update_normals_block_indices_host_ = block_indices;
  update_normals_block_indices_device_ = update_normals_block_indices_host_;

  // View of BlockIndices to TsdfBlock potiners. We need this to do global
  // position lookups for all voxels.
  const GPULayerView<TsdfBlock> gpu_layer = tsdf_layer.getGpuLayerView();

  tex::updateTexVoxelDirectionsGPU(
      gpu_layer, update_normals_tex_block_prts_device_,
      update_normals_block_indices_device_, num_blocks, integration_stream_,
      tsdf_layer.block_size(), tsdf_layer.voxel_size());
}

}  // namespace nvblox

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

  timing::Timer update_neighbor_block_indices_timer(
      "tex/integrate/neighbor_block_indices");
  updateNeighborIndicies(tsdf_layer, block_indices);
  update_neighbor_block_indices_timer.Stop();

  tex_layer->waitForPrefetch();
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
               camera, truncation_distance_m, tsdf_layer, tex_layer);
  update_blocks_timer.Stop();

  if (updated_blocks != nullptr) {
    *updated_blocks = block_indices;
  }
}

void ProjectiveTexIntegrator::updateNeighborIndicies(
    const TsdfLayer& tsdf_layer, const std::vector<Index3D>& block_indices) {
  const int new_size = block_indices.size() * tex::neighbor::kCubeNeighbors;
  tsdf_block_ptrs_host_.resize(new_size);
  tsdf_block_ptrs_device_.resize(new_size);

  for (int i = 0; i < block_indices.size(); ++i) {
    for (int j = 0; j < tex::neighbor::kCubeNeighbors; ++j) {
      Index3D offset = tex::neighbor::blockOffsetFromNeighborIndex(j);
      tsdf_block_ptrs_host_[i * tex::neighbor::kCubeNeighbors + j] =
          tsdf_layer.getBlockAtIndex(block_indices[i] + offset).get();
    }
  }
  tsdf_block_ptrs_device_ = tsdf_block_ptrs_host_;
}

__device__ float computeMeasurementWeight(const TexVoxel* tex_voxel,
                                          const Transform& T_C_L,
                                          const Vector3f& voxel_center,
                                          const Vector2f& u_px,
                                          const float u_px_depth) {
  // Area based weighting
  // Minimum depth of scanning (m). I.e. closest we will get to a point.
  constexpr float MIN_DEPTH = .1f;
  // Smoothing for the deviation in normal direction we accept for w_area
  constexpr float SIMGA_AREA = 2.f;
  // Smoothing for the deviation in normal direction we accept for w_angle
  constexpr float SIMGA_ANGLE = 1.f;
  constexpr float MIN_W_AREA = .1f;   // GAMAM_AREA in TextureFusion Paper
  constexpr float MIN_W_ANGLE = .1f;  // GAMAM_ANGLE in TextureFusion Paper

  Vector3f view_dir = (T_C_L.translation() - voxel_center).normalized();
  float normal_align =
      tex::texDirToWorldVector(tex_voxel->dir).dot(view_dir);  // in [-1, 1]
  float depth_clipped = fmax(u_px_depth, MIN_DEPTH);
  // rho is the product of the alignment of the view direction with the surface
  // normal at the given voxel and the clipped inverse depth. I.e. voxels that
  // we look at head on and are close to the camera are preferred
  float rho =
      powf(MIN_DEPTH / depth_clipped, 2.f) * normal_align;  // in [-1, 1]

  // w_area is a bell curve centered at 1 with rho as a parameter. So the closer
  // rho is to one, the more weight we assign it. SIGMA_AREA controlls the
  // sharpness of the falloff at around the mean 1.
  // clang-format off
  float w_area = fmax(
      expf(-powf((1 - rho) / SIMGA_AREA, 2.f)), 
      MIN_W_AREA
  ); // in [MIN_W_AREA, 1]
  // clang-format on

  // View angle based weighting
  // clang-format off
  float w_angle = fmax(
    expf(-powf((1 - normal_align) / SIMGA_ANGLE, 2.f)),
    MIN_W_ANGLE
  ); // in [MIN_W_ANGLE, 1]
  // clang-format on

  return w_area * w_angle;  // in [0, 1]
}

__device__ inline void updateTexel(const Color& color_measured,
                                   TexVoxel* tex_voxel,
                                   const Index2D& texel_idx,
                                   const float measurement_weight,
                                   const float max_weight) {
  tex_voxel->color(texel_idx) =
      blendTwoColors(tex_voxel->color(texel_idx), tex_voxel->weight,
                     color_measured, measurement_weight);
}

__device__ const TsdfVoxel* getNeighborVoxelAtIndex(
    const TsdfBlock** blocks, const Index3D& voxel_index) {
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
  // get the neighboring voxel either from a neighboring block if it's outside
  // the current one, or get it directly from the current block
  int linear_neighbor_idx =
      tex::neighbor::neighborBlockIndexFromOffset(block_offset);
  const TsdfBlock* block =
      blocks[block_idx * tex::neighbor::kCubeNeighbors + linear_neighbor_idx];

  if (block == nullptr) {
    return nullptr;
  }
  return &block->voxels[voxel_idx.x()][voxel_idx.y()][voxel_idx.z()];
}

/**
 * @brief Gets TSDF values at two positions at the same time. Where the surface
 * is sampled around position1 and extrapolated to position2.
 * This saves time, compared to two separate interpolations
 *
 * @param blocks neihbor blocks
 * @param position1 first position to interpolate the surface distance at
 * @param position2 second position to interpolate the surface distance at
 * @param index1 voxel index of position1
 * @param voxel_size
 * @param sdf1 output sdf at position1
 * @param sdf2 output sdf at position2
 * @return __device__ if the interpolation was successful
 */
__device__ bool getTSDFValues(const TsdfBlock** blocks,
                              const Vector3f& position1,
                              const Vector3f& position2, const Index3D& index1,
                              const float voxel_size, float* sdf1,
                              float* sdf2) {
  Vector3f normalized_pos1 = position1 / voxel_size;
  Vector3f normalized_pos2 = position2 / voxel_size;
  // SDF interpolation Weights are in range [0,1] for each axis
  Vector3f weight1 = normalized_pos1.array() - normalized_pos1.array().floor();
  Vector3f weight2 = normalized_pos2.array() - normalized_pos1.array().floor();
  // clear sdf values
  (*sdf1) = 0.f;
  (*sdf2) = 0.f;

  const TsdfVoxel* v;
  // clang-format off
  v = getNeighborVoxelAtIndex(blocks, index1);                    if (v == nullptr) return false;     (*sdf1) += (1.0f - weight1.x()) * (1.0f - weight1.y()) * (1.0f - weight1.z()) * v->distance; 
                                                                                                      (*sdf2) += (1.0f - weight2.x()) * (1.0f - weight2.y()) * (1.0f - weight2.z()) * v->distance; 
  v = getNeighborVoxelAtIndex(blocks, index1 + Index3D(1, 0, 0)); if (v == nullptr) return false;     (*sdf1) += weight1.x()          * (1.0f - weight1.y()) * (1.0f - weight1.z()) * v->distance; 
                                                                                                      (*sdf2) += weight2.x()          * (1.0f - weight2.y()) * (1.0f - weight2.z()) * v->distance; 
  v = getNeighborVoxelAtIndex(blocks, index1 + Index3D(0, 1, 0)); if (v == nullptr) return false;     (*sdf1) += (1.0f - weight1.x()) * weight1.y()          * (1.0f - weight1.z()) * v->distance; 
                                                                                                      (*sdf2) += (1.0f - weight2.x()) * weight2.y()          * (1.0f - weight2.z()) * v->distance; 
  v = getNeighborVoxelAtIndex(blocks, index1 + Index3D(0, 0, 1)); if (v == nullptr) return false;     (*sdf1) += (1.0f - weight1.x()) * (1.0f - weight1.y()) * weight1.z()          * v->distance; 
                                                                                                      (*sdf2) += (1.0f - weight2.x()) * (1.0f - weight2.y()) * weight2.z()          * v->distance; 
  v = getNeighborVoxelAtIndex(blocks, index1 + Index3D(1, 1, 0)); if (v == nullptr) return false;     (*sdf1) += weight1.x()          * weight1.y()          * (1.0f - weight1.z()) * v->distance; 
                                                                                                      (*sdf2) += weight2.x()          * weight2.y()          * (1.0f - weight2.z()) * v->distance; 
  v = getNeighborVoxelAtIndex(blocks, index1 + Index3D(0, 1, 1)); if (v == nullptr) return false;     (*sdf1) += (1.0f - weight1.x()) * weight1.y()          * weight1.z()          * v->distance; 
                                                                                                      (*sdf2) += (1.0f - weight2.x()) * weight2.y()          * weight2.z()          * v->distance; 
  v = getNeighborVoxelAtIndex(blocks, index1 + Index3D(1, 0, 1)); if (v == nullptr) return false;     (*sdf1) += weight1.x()          * (1.0f - weight1.y()) * weight1.z()          * v->distance; 
                                                                                                      (*sdf2) += weight2.x()          * (1.0f - weight2.y()) * weight2.z()          * v->distance; 
  v = getNeighborVoxelAtIndex(blocks, index1 + Index3D(1, 1, 1)); if (v == nullptr) return false;     (*sdf1) += weight1.x()          * weight1.y()          * weight1.z()          * v->distance; 
                                                                                                      (*sdf2) += weight2.x()          * weight2.y()          * weight2.z()          * v->distance;
  // clang-format on
  return true;
}

/**
 * @brief For the line between (x1, y1), (x2, y2) we find the intersection with
 * the x-axis.
 * Solves y - y1 = (y2 - y1) / (x2 - x1) * (x - x1) = 0 for x
 *
 * @param x1 x coordinate of point1
 * @param x2 x coordinate of point2
 * @param y1 y coordinate of point1
 * @param y2 y coordinate of point2
 * @return __device__
 */
__device__ inline float findIntersectionLinear(const float x1, const float x2,
                                               const float y1, const float y2) {
  return x1 + (y1 / (y1 - y2)) * (x2 - x1);
}

/**
 * @brief Finds the intersection of the ray starting at the given postion along
 * the given direction with the TSDF surface. The distance along this ray is
 * returned.
 *
 * @param blocks neighbor blocks
 * @param voxel_size voxel size
 * @param position stating position
 * @param direction ray directions
 * @param distance dinstance from position along dir until the intersection
 * @return __device__ if an intersection could be found along the ray
 */
__device__ bool raycastToSurface(const TsdfBlock** blocks,
                                 const float voxel_size,
                                 const Vector3f& position,
                                 const TexVoxel::Dir& direction,
                                 float* distance) {
  const Index3D voxel_idx = Index3D(threadIdx.z, threadIdx.y, threadIdx.x);
  Vector3f next_voxel = tex::texDirToWorldVector(direction) * voxel_size;
  float sdf1, sdf2;

  if (getTSDFValues(blocks, position, position + next_voxel, voxel_idx,
                    voxel_size, &sdf1, &sdf2)) {
    (*distance) = findIntersectionLinear(0.f, voxel_size, sdf1, sdf2);
    return true;
  }
  return false;
}

__global__ void integrateBlocks(
    const Index3D* block_indices_device_ptr, const Camera camera,
    const Color* color_image, const int color_rows, const int color_cols,
    const float* depth_image, const int depth_rows, const int depth_cols,
    const Transform T_C_L, const float block_size,
    const float truncation_distance_m, const float max_weight,
    const float max_integration_distance, const int depth_subsample_factor,
    const TsdfBlock** tsdf_blocks, TexBlock** tex_ptrs) {
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
  const Vector2f u_px_depth = u_px / static_cast<float>(depth_subsample_factor);
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
  TexVoxel* voxel_ptr =
      &(tex_ptrs[blockIdx.x]->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // NOTE(rasaford): If the current voxel has not been assigned a texture plane
  // direction, it must not be on the truncation band --> skip it
  if (!voxel_ptr->isInitialized()) {
    return;
  }

  // Update the weight of each tex voxel once per voxel (instead of once per
  // texel) as the average of the new and old weights
  const Index3D& block_idx = block_indices_device_ptr[blockIdx.x];
  const Index3D voxel_idx = Index3D(threadIdx.z, threadIdx.y, threadIdx.x);
  const Vector3f voxel_center = getCenterPostionFromBlockIndexAndVoxelIndex(
      block_size, block_idx, voxel_idx);

  // float measurement_weight = 1.f;

  float measurement_weight = computeMeasurementWeight(
      voxel_ptr, T_C_L, voxel_center, u_px, surface_depth_m);
  Color image_value = Color::Black();
  Index2D texel_idx{0, 0};
  Vector3f texel_pos = Vector3f::Zero();
  Vector3f surface_pos = Vector3f::Zero();
  for (int row = 0; row < voxel_ptr->kPatchWidth; ++row) {
    for (int col = 0; col < voxel_ptr->kPatchWidth; ++col) {
      texel_idx = Index2D(row, col);
      image_value = Color::Black();

      // Orthogonal projection of TexVoxel tile to SDF surface
      texel_pos = getCenterPositionForTexel(block_size, block_idx, voxel_idx,
                                            texel_idx, voxel_ptr->dir);

      float distance;
      if (!raycastToSurface(tsdf_blocks, block_size / TsdfBlock::kVoxelsPerSide,
                            texel_pos, voxel_ptr->dir, &distance)) {
        continue;
      }

      surface_pos =
          texel_pos + distance * tex::texDirToWorldVector(voxel_ptr->dir);

      // Project the current texel_idx to image space. If it's outside the
      // image, go to the next texel.
      surface_pos = T_C_L * surface_pos;
      if (!camera.project(surface_pos, &u_px)) {
        continue;
      }
      // sample the color at the interpolated point
      if (!interpolation::interpolate2DLinear<Color>(
              color_image, u_px, color_rows, color_cols, &image_value)) {
        continue;
      }
      // update the texel color
      updateTexel(image_value, voxel_ptr, texel_idx, measurement_weight,
                  max_weight);
    }
  }
  // Since the voxel_weight is read when updating the texels, it must be updated
  // after all texels. This is a non-saturating filter version of the weighting
  // rule described in TextureFusion
  voxel_ptr->weight = (measurement_weight + voxel_ptr->weight) / 2;
}

void ProjectiveTexIntegrator::updateBlocks(
    const std::vector<Index3D>& block_indices, const ColorImage& color_frame,
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    const float truncation_distance_m, const TsdfLayer& tsdf_layer,
    TexLayer* tex_layer_ptr) {
  CHECK_NOTNULL(tex_layer_ptr);
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
    tex_block_ptrs_device_.reserve(new_size);
    block_indices_host_.reserve(new_size);
    tex_block_ptrs_host_.reserve(new_size);
  }

  // Stage on the host pinned memory
  block_indices_host_ = block_indices;
  tex_block_ptrs_host_ = getBlockPtrsFromIndices(block_indices, tex_layer_ptr);

  // Transfer to the device
  block_indices_device_ = block_indices_host_;
  tex_block_ptrs_device_ = tex_block_ptrs_host_;

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
      tex_layer_ptr->block_size(),
      truncation_distance_m,
      max_weight_,
      max_integration_distance_m_,
      depth_subsampling_factor,
      tsdf_block_ptrs_device_.data(),
      tex_block_ptrs_device_.data());
  // clang-format on
  checkCudaErrors(cudaStreamSynchronize(integration_stream_));
  checkCudaErrors(cudaPeekAtLastError());

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
  if (block_indices.empty()) {
    return;
  }
  // Get the pointers for the indexed blocks from both
  // - The tsdf layer: Since all Voxels are already integrated here, we read
  // from this layer to estimate the normal direcitonA
  // - The TexBlock layer: We write the updated directions to this layer for
  // all new blocks
  // NOTE(rasaford) Even though we do not modify TsdfLayer, we have to drop
  // const here due to the way the GPUHashMap works
  // TsdfLayer* tsdf_layer_non_const = const_cast<TsdfLayer*>(&tsdf_layer);
  // std::vector<TsdfBlock*> tsdf_block_ptrs =
  //     getBlockPtrsFromIndices(block_indices, tsdf_layer_non_const);
  std::vector<TexBlock*> tex_block_ptrs =
      getBlockPtrsFromIndices(block_indices, tex_layer_ptr);

  // // We assume that a TsdfBlock at index i corresponds to a TexBlock at i.
  // This
  // // cannot be the case if the two vectors don't have the same number of
  // // elements
  // CHECK_EQ(tsdf_block_ptrs.size(), tex_block_ptrs.size());

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

  tex::updateTexVoxelDirectionsGPU(
      tsdf_block_ptrs_device_, update_normals_block_indices_device_,
      update_normals_tex_block_prts_device_, num_blocks, integration_stream_,
      tsdf_layer.block_size(), tsdf_layer.voxel_size());
}

}  // namespace nvblox

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
               camera, truncation_distance_m, tsdf_layer, tex_layer);
  update_blocks_timer.Stop();

  if (updated_blocks != nullptr) {
    *updated_blocks = block_indices;
  }
}

__device__ inline float computeMeasurementWeight(const TexVoxel* tex_voxel,
                                                 const Transform T_C_L,
                                                 const Vector3f& voxel_center) {
  // TODO: (rasaford) compute measurement weight based on e.g.
  // - size of the projected texel in the image
  // - sharpness of the projected area in the image (to compensate motion blur)
  // - how flat on we're looking at the texel projection
  // - if the texel is on a boundary
  // - ...

  // NOTE(rasaford) the resulting world vector is always normlaized
  Vector3f world_vector = tex::texDirToWorldVector(tex_voxel->dir);

  Vector3f view_dir = (T_C_L.translation() - voxel_center).normalized();

  return fabs(world_vector.dot(view_dir));
}

__device__ inline void updateTexel(const Color& color_measured,
                                   TexVoxel* tex_voxel,
                                   const Index2D& texel_idx,
                                   const float measurement_weight) {
  const Color old_color = (*tex_voxel)(texel_idx);
  (*tex_voxel)(texel_idx) = blendTwoColors(old_color, tex_voxel->weight,
                                           color_measured, measurement_weight);
}

__device__ const TsdfVoxel* getNeighboringVoxel(const TsdfBlock** blocks,
                                                const Index3D& voxel_index,
                                                const Index3D& voxel_offset) {
  const int block_idx = blockIdx.x;
  Index3D neighbor_voxel_idx = voxel_index + voxel_offset;

  Index3D block_offset{0, 0, 0};
  for (int i = 0; i < 3; ++i) {
    if (neighbor_voxel_idx[i] >= TsdfBlock::kVoxelsPerSide) {
      neighbor_voxel_idx[i] -= TsdfBlock::kVoxelsPerSide;
      block_offset[i] = 1;
    } else if (neighbor_voxel_idx[i] < 0) {
      neighbor_voxel_idx[i] += TsdfBlock::kVoxelsPerSide;
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
  return &block->voxels[neighbor_voxel_idx.x()][neighbor_voxel_idx.y()]
                       [neighbor_voxel_idx.z()];
}

__device__ bool bilinearInterpolation(
    const TsdfBlock** blocks, const Index3D& voxel_index,
    const Index3D& offset1, const Index3D& offset2, const Vector3f& position,
    const Vector3f& voxel_position, const float voxel_size, float* sdf) {
  // the voxels are named in the following way regarding to the given base
  // voxel_index (v0)
  // clang-format off
  // v3 ---------- v1
  // |              |
  // |              |
  // |              dir1
  // |              |
  // |              |
  // v2 -- dir2 -- v0
  // clang-format on

  // clang-format off
  const TsdfVoxel* v0 = getNeighboringVoxel(blocks, voxel_index, Index3D(0, 0, 0));
  const TsdfVoxel* v1 = getNeighboringVoxel(blocks, voxel_index, offset1);
  const TsdfVoxel* v2 = getNeighboringVoxel(blocks, voxel_index, offset2);
  const TsdfVoxel* v3 = getNeighboringVoxel(blocks, voxel_index, offset1 + offset2);
  // clang-fromat on

  if (v0 == nullptr || v1 == nullptr || v2 == nullptr || v3 == nullptr) {
    return false;
  }

  const Vector3f v1_pos = voxel_position + voxel_size * offset1.cast<float>();
  const Vector3f v2_pos = voxel_position + voxel_size * offset2.cast<float>();

  Vector3f position_dir = position - voxel_position;
  float d1 = position_dir.dot(v1_pos - voxel_position);
  float d2 = position_dir.dot(v2_pos - voxel_position);
  // printf("d1: %f, d2: %f, voxel_size: %f\n", d1, d2, voxel_size);
  // clang-format off
  // interpolate first along direction1
  float sdf1_1 = d1 / voxel_size * v0->distance + (voxel_size - d1) / voxel_size * v1->distance;
  float sdf1_2 = d1 / voxel_size * v2->distance + (voxel_size - d1) / voxel_size * v3->distance;
  // clang-format on
  // interpolation along direction2
  *sdf = d2 / voxel_size * sdf1_1 + (voxel_size - d2) / voxel_size * sdf1_2;
  return true;
}

__device__ bool trilinearSurfaceInterpolation(
    const TsdfBlock** blocks, const float voxel_size, const Vector3f& position,
    const Vector3f& voxel_center, const TexVoxel::Dir& direction,
    const float block_size, float* distance) {
  const Index3D voxel_idx = Index3D(threadIdx.z, threadIdx.y, threadIdx.x);

  // Get the current voxel via the accessing function, since we need to convert
  // to a linear index to get access to the current block
  const TsdfVoxel* current_voxel =
      getNeighboringVoxel(blocks, voxel_idx, Index3D(0, 0, 0));

  if (current_voxel == nullptr) {
    // printf("current voxel null \n");
    return false;
  }

  TexVoxel::Dir dir = direction;
  float dir_factor = 1.f;
  Index3D dir_vec = tex::texDirToWorldVector(dir).cast<int>();

  const TsdfVoxel* neighbor_voxel =
      getNeighboringVoxel(blocks, voxel_idx, dir_vec);

  // if the current and neighboring voxel are on the same side of the SDF
  // boundary, seach in the other direction to find the SDF boundary
  if (neighbor_voxel == nullptr ||
      tex::isSameSign<float>(current_voxel->distance,
                             neighbor_voxel->distance)) {
    dir_vec = -dir_vec;
    dir_factor = -1.f;
    dir = tex::negateDir(dir);
    neighbor_voxel = getNeighboringVoxel(blocks, voxel_idx, dir_vec);

    // If the SDF did not change sign in any direction along the given dir, the
    // current voxel cannot be zero crossing along it.
    if (neighbor_voxel == nullptr ||
        tex::isSameSign<float>(current_voxel->distance,
                               neighbor_voxel->distance)) {
      return false;
    }
  }

  // linear interpolation along each axis
  Vector3f pos_dir = position - voxel_center;
  Vector3f pos_offset = voxel_size * dir_vec.cast<float>();

  Index3D offset1, offset2;
  switch (dir) {
    case TexVoxel::Dir::X_PLUS:
    case TexVoxel::Dir::X_MINUS:
      offset1 = pos_dir[1] > 0 ? Index3D(0, 1, 0) : Index3D(0, -1, 0);
      offset2 = pos_dir[2] > 0 ? Index3D(0, 0, 1) : Index3D(0, 0, -1);
      break;
    case TexVoxel::Dir::Y_PLUS:
    case TexVoxel::Dir::Y_MINUS:
      offset1 = pos_dir[0] > 0 ? Index3D(1, 0, 0) : Index3D(-1, 0, 0);
      offset2 = pos_dir[2] > 0 ? Index3D(0, 0, 1) : Index3D(0, 0, -1);
      break;
    case TexVoxel::Dir::Z_PLUS:
    case TexVoxel::Dir::Z_MINUS:
      offset1 = pos_dir[0] > 0 ? Index3D(1, 0, 0) : Index3D(-1, 0, 0);
      offset2 = pos_dir[1] > 0 ? Index3D(0, 1, 0) : Index3D(0, -1, 0);
      break;
    default:
      return false;
  }

  float sdf1 = 0.f, sdf2 = 0.f;
  bool valid = bilinearInterpolation(blocks, voxel_idx, offset1, offset2,
                                     position, voxel_center, voxel_size, &sdf1);
  valid &= bilinearInterpolation(blocks, voxel_idx + dir_vec, offset1, offset2,
                                 position + pos_offset,
                                 voxel_center + pos_offset, voxel_size, &sdf2);
  // if the current voxel does not have enough neighboring voxels to perform
  // each 2D interpolation, the resulting sdf valus are invalid.
  if (!valid) return false;
  if (tex::isSameSign<float>(sdf1, sdf2)) return false;

  // printf("stf1: %f, stf2: %f\n", sdf1, sdf2);
  // TODO: compute this based on distance to the surface
  *distance = voxel_size * dir_factor * fabs(sdf1) / (fabs(sdf1) + fabs(sdf2));
  return true;
}

__global__ void integrateBlocks(
    const Index3D* block_indices_device_ptr, const Camera camera,
    const Color* color_image, const int color_rows, const int color_cols,
    const float* depth_image, const int depth_rows, const int depth_cols,
    const Transform T_C_L, const float block_size,
    const float truncation_distance_m, const float max_weight,
    const float max_integration_distance, const int depth_subsample_factor,
    const TsdfBlock** tsdf_blocks, TexBlock** tex_ptrs) {
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
  Eigen::Vector2f u_px_depth =
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

  float measurement_weight =
      computeMeasurementWeight(voxel_ptr, T_C_L, voxel_center);

  Color image_value = Color::Black();
  Index2D texel_idx{0, 0};
  Vector3f texel_pos = Vector3f::Zero();
  Vector3f surface_pos = Vector3f::Zero();
  // loop over all colors in the TexVoxel patch
  for (int row = 0; row < voxel_ptr->kPatchWidth; ++row) {
    for (int col = 0; col < voxel_ptr->kPatchWidth; ++col) {
      texel_idx = Index2D(row, col);
      image_value = Color::Black();

      // Orthogonal projection of texVoxel Tile to SDF surface
      texel_pos = getCenterPositionForTexel(block_size, block_idx, voxel_idx,
                                            texel_idx, voxel_ptr->dir);
      float distance;
      if (!trilinearSurfaceInterpolation(
              tsdf_blocks, block_size / TsdfBlock::kVoxelsPerSide, texel_pos,
              voxel_center, voxel_ptr->dir, block_size, &distance)) {
        continue;
      }
      printf("distance: %f\n", distance);
      surface_pos =
          texel_pos + distance * tex::texDirToWorldVector(voxel_ptr->dir);

      // Project the current texel_idx to image space. If it's outside the
      // image, go to the next texel.
      surface_pos = T_C_L * surface_pos;
      if (!camera.project(surface_pos, &u_px)) {
        continue;
      }

      if (!interpolation::interpolate2DLinear<Color>(
              color_image, u_px, color_rows, color_cols, &image_value))
        continue;

      updateTexel(image_value, voxel_ptr, texel_idx, measurement_weight);
    }
  }
  // Since the voxel_weight is read when updating the texels, it must be updated
  // after all texels
  voxel_ptr->weight =
      fmin(fmax(measurement_weight, voxel_ptr->weight), max_weight);
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

  tsdf_block_ptrs_host_.resize(block_indices.size() *
                               tex::neighbor::kCubeNeighbors);
  tsdf_block_ptrs_device_.resize(block_indices.size() *
                                 tex::neighbor::kCubeNeighbors);
  for (int i = 0; i < block_indices.size(); ++i) {
    for (int j = 0; j < tex::neighbor::kCubeNeighbors; ++j) {
      Index3D offset = tex::neighbor::blockOffsetFromNeighborIndex(j);
      tsdf_block_ptrs_host_[i * tex::neighbor::kCubeNeighbors + j] =
          tsdf_layer.getBlockAtIndex(block_indices[i] + offset).get();
    }
  }
  tsdf_block_ptrs_device_ = tsdf_block_ptrs_host_;

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
  // std::cout << "update done" << std::endl;

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
  const GPULayerView<TsdfBlock> tsdf_layer_view = tsdf_layer.getGpuLayerView();
  const GPULayerView<TexBlock> tex_layer_view =
      tex_layer_ptr->getGpuLayerView();

  tex::updateTexVoxelDirectionsGPU(
      tsdf_layer_view, tex_layer_view, update_normals_tex_block_prts_device_,
      update_normals_block_indices_device_, num_blocks, integration_stream_,
      tsdf_layer.block_size(), tsdf_layer.voxel_size());
}

}  // namespace nvblox

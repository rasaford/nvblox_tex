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

#include <Eigen/Core>

#include "nvblox/core/color.h"
#include "nvblox/core/types.h"

namespace nvblox {

struct TsdfVoxel {
  // Signed projective distance of the voxel from a surface.
  float distance = 0.0f;
  // How many observations/how confident we are in this observation.
  float weight = 0.0f;
};

struct EsdfVoxel {
  // TODO(helen): optimize the memory layout here.
  // Cached squared distance towards the parent.
  float squared_distance_vox = 0.0f;
  // Direction towards the parent, *in units of voxels*.
  Eigen::Vector3i parent_direction = Eigen::Vector3i::Zero();
  // Whether this voxel is inside the surface or not.
  bool is_inside = false;
  // Whether this voxel has been observed.
  bool observed = false;
  // Whether this voxel is a "site": i.e., near the zero-crossing and is
  // eligible to be considered a parent.
  bool is_site = false;
};

struct ColorVoxel {
  Color color = Color::Gray();
  // How many observations/how confident we are in this observation.
  float weight = 0.0f;
};

// Each TexVoxel is a 2d grid of colors for each voxel
template <typename _ElementType, int _PatchWidth>
class TexVoxelTemplate {
  // Six possible directors for each 2D texture plane to be facing in
  // 3D space. The orientation of each plane will be determined at allocation
  // time
 public:
  enum class Dir { X_PLUS, X_MINUS, Y_PLUS, Y_MINUS, Z_PLUS, Z_MINUS, NONE };
  // number of elements in the enum has to always match
  static constexpr int Dir_count = 7;

  // make the template params queryable as TexVoxelTemplate::ElementType etc.
  typedef _ElementType ElementType;

  __host__ __device__ inline bool isInitialized() const {
    return dir != Dir::NONE;
  }

  __host__ __device__ inline void updateDir(const Dir dir,
                                            const float dir_weight) {
    this->dir = dir;
    this->dir_weight = dir_weight;
    this->weight = 0.f;
  }

  // Access
  __host__ __device__ inline ElementType operator()(const int row_idx,
                                                    const int col_idx) const {
    return colors[row_idx * kPatchWidth + col_idx];
  }
  __host__ __device__ inline ElementType& operator()(const int row_idx,
                                                     const int col_idx) {
    return colors[row_idx * kPatchWidth + col_idx];
  }
  __host__ __device__ inline ElementType operator()(const Index2D& idx) const {
    return colors[idx(0) * kPatchWidth + idx(1)];
  }
  __host__ __device__ inline ElementType& operator()(const Index2D& idx) {
    return colors[idx(0) * kPatchWidth + idx(1)];
  }
  __host__ __device__ inline ElementType operator()(
      const int linear_idx) const {
    return colors[linear_idx];
  }
  __host__ __device__ inline ElementType& operator()(const int linear_idx) {
    return colors[linear_idx];
  }

  // hopefully this value does not get stored for every TexVoxel this way
  static constexpr int kPatchWidth = _PatchWidth;
  // patch size in pixels. Each TexVoxel is of size (kPatchWidth, kPatchWidth)
  // NOTE(rasaford) We intentionally chose an array of colors here, instead of
  // using the functionality in image.h, since we want to use the absolute
  // minimum amount of memory for each TexVoxel.
  Color colors[kPatchWidth * kPatchWidth];
  // how confident we are in the colors for this observation
  float weight = 0.;

  // any of six asis aligned directions the 2d texture plane is facing in 3d
  // space
  Dir dir = Dir::NONE;
  // how confident we are in the direction estimate for this observation
  float dir_weight = 0.;

  // static parameter
  static constexpr float DIR_THRESHOLD = 0.9f;
};

// For convenience we define this non-templated version of TexVoxel.
// Each TexVoxel is a 2d grid of colors for each voxel
// NOTE(rasaford) To actually be able to use this type, it needs to be defined
// in voxels.cpp (above its only the declaration).
#ifdef TEXEL_SIZE
typedef TexVoxelTemplate<Color, TEXEL_SIZE> TexVoxel;
#else
typedef TexVoxelTemplate<Color, 8> TexVoxel;
#endif

struct FreespaceVoxel {
  bool free = true;
};

}  // namespace nvblox

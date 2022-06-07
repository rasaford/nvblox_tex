#pragma once

#include "nvblox/core/common_names.h"
#include "nvblox/core/types.h"
#include "nvblox/core/voxels.h"

namespace nvblox {
namespace tex {

/**
 * @brief Convert the given TexVoxel::Dir to a normalized vector
 *
 * @param dir
 * @return __host__
 */
__host__ __device__ inline Vector3f texDirToWorldVector(TexVoxel::Dir dir) {
  Vector3f world_dir;
  switch (dir) {
    case TexVoxel::Dir::X_PLUS:
      world_dir << 1.0f, 0.0f, 0.0f;
      break;
    case TexVoxel::Dir::X_MINUS:
      world_dir << -1.0f, 0.0f, 0.0f;
      break;
    case TexVoxel::Dir::Y_PLUS:
      world_dir << 0.0f, 1.0f, 0.0f;
      break;
    case TexVoxel::Dir::Y_MINUS:
      world_dir << 0.0f, -1.0f, 0.0f;
      break;
    case TexVoxel::Dir::Z_PLUS:
      world_dir << 0.0f, 0.0f, 1.0f;
      break;
    case TexVoxel::Dir::Z_MINUS:
      world_dir << 0.0f, 0.0f, -1.0f;
      break;
  }
  return world_dir;
};

/**
 * @brief Get an othogoanal direction for a given TexVoxel::Dir.
 *
 * @param dir direction
 * @param positive if the orthogonal direction should be the positive one,
 * default: true
 */
__host__ __device__ inline TexVoxel::Dir orthogonalDir(
    const TexVoxel::Dir dir, const bool positive = true) {
  switch (dir) {
    case TexVoxel::Dir::X_PLUS:
    case TexVoxel::Dir::X_MINUS:
      return positive ? TexVoxel::Dir::Y_PLUS : TexVoxel::Dir::Y_MINUS;
    case TexVoxel::Dir::Y_PLUS:
    case TexVoxel::Dir::Y_MINUS:
      return positive ? TexVoxel::Dir::Z_PLUS : TexVoxel::Dir::Z_MINUS;
    case TexVoxel::Dir::Z_PLUS:
    case TexVoxel::Dir::Z_MINUS:
      return positive ? TexVoxel::Dir::X_PLUS : TexVoxel::Dir::X_MINUS;
  }
}

// Negation operator. Inverses the given direction
__host__ __device__ inline TexVoxel::Dir negateDir(const TexVoxel::Dir& dir) {
  switch (dir) {
    // clang-format off
    case TexVoxel::Dir::X_PLUS:   return TexVoxel::Dir::X_MINUS;
    case TexVoxel::Dir::X_MINUS:  return TexVoxel::Dir::X_PLUS;
    case TexVoxel::Dir::Y_PLUS:   return TexVoxel::Dir::Y_MINUS;
    case TexVoxel::Dir::Y_MINUS:  return TexVoxel::Dir::Y_PLUS;
    case TexVoxel::Dir::Z_PLUS:   return TexVoxel::Dir::Z_MINUS;
    case TexVoxel::Dir::Z_MINUS:  return TexVoxel::Dir::Z_PLUS;
    default:                      return TexVoxel::Dir::NONE;
      // clang-format on
  }
}

}  // namespace tex
}  // namespace nvblox
#pragma once

#include "nvblox/core/common_names.h"
#include "nvblox/core/types.h"
#include "nvblox/core/voxels.h"

namespace nvblox {
namespace tex {

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

}  // namespace tex
}  // namespace nvblox
#pragma once

#include <Eigen/Core>
#include "nvblox/core/common_names.h"
#include "nvblox/core/voxels.h"
#include "nvblox/tex/tex_conversions.h"

namespace nvblox {
namespace tex {
/**
 * @brief Tests if the given numeric elements have the same sign
 */
template <typename T>
__host__ __device__ inline bool isSameSign(const T a, const T b) {
  return a * b >= 0.f;
}

namespace neighbor {

static constexpr int kCubeNeighbors = 27;
/**
 * @brief Computes the linear index from a 3D block offset
 *
 * @param block_offset block offset x, where each coordinate x_i is -1 <= x_i <=
 * 1 (at most 1 away from the origin in each direction)
 * @return __device__
 */
__host__ __device__ inline int neighborBlockIndexFromOffset(
    const Index3D& block_offset) {
  return Index3D(9, 3, 1).dot(block_offset + Index3D(1, 1, 1));
}

__host__ __device__ inline Index3D blockOffsetFromNeighborIndex(
    const int linear_index) {
  return Index3D(linear_index / 9, (linear_index % 9) / 3, linear_index % 3) -
         Index3D(1, 1, 1);
}

}  // namespace neighbor
}  // namespace tex
}  // namespace nvblox

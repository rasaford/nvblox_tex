#pragma once

#include <Eigen/Core>
#include <vector>
#include "nvblox/core/common_names.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/tex/surface_estimation.h"
#include "nvblox/tex/tex_conversions.h"

namespace nvblox {
namespace tex {

typedef Eigen::Matrix<float, 7, 1> Vector7f;
typedef Eigen::Matrix<int, 7, 1> Vector7i;

void updateTexVoxelDirectionsGPU(
    device_vector<const TsdfBlock*> neighbor_blocks,
    device_vector<TexBlock*>& tex_block_ptrs, const int num_blocks,
    const cudaStream_t stream, const float block_size, const float voxel_size);

}  // namespace tex
}  // namespace nvblox

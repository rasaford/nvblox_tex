#pragma once

#include <Eigen/Core>
#include <vector>
#include "nvblox/core/common_names.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/tex/tex_conversions.h"

namespace nvblox {
namespace tex {

typedef Eigen::Matrix<float, 7, 1> Vector7f;

void updateTexVoxelDirectionsGPU(
    const GPULayerView<TsdfBlock>& tsdf_layer_view,
    const GPULayerView<TexBlock>& tex_layer_view,
        device_vector<TexBlock*>& tex_block_ptrs,
    const device_vector<Index3D>& block_indices_device, const int num_blocks,
    const cudaStream_t stream, const float block_size, const float voxel_size);

}  // namespace tex
}  // namespace nvblox

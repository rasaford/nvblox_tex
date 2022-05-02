#pragma once

#include <vector>
#include "nvblox/core/common_names.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"

namespace nvblox {
namespace tex {

void updateTexVoxelDirectionsGPU(
    const GPULayerView<TsdfBlock> gpu_layer,
    device_vector<TexBlock*>& tex_block_ptrs,
    const device_vector<Index3D>& block_indices_device, const int num_blocks,
    const cudaStream_t stream, const float block_size, const float voxel_size);

}  // namespace tex
}  // namespace nvblox

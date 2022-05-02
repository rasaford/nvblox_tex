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

#include "nvblox/core/blox.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/types.h"
#include "nvblox/core/voxels.h"
#include "nvblox/mesh/mesh_block.h"

namespace nvblox {

using TsdfBlock = VoxelBlock<TsdfVoxel>;
using TsdfLayer = VoxelBlockLayer<TsdfVoxel>;
using EsdfBlock = VoxelBlock<EsdfVoxel>;
using EsdfLayer = VoxelBlockLayer<EsdfVoxel>;

// TODO (rasaford) For TexVoxels Blocks don't make much sense as a grouping abstraction,
// since we already have a grid of pixels within each texvoxel. (for thread level parallelism)
// Instead the block abstraction should be skipped here.
using ColorBlock = VoxelBlock<ColorVoxel>;
using ColorLayer = VoxelBlockLayer<ColorVoxel>;
using TexBlock = VoxelBlock<TexVoxel>;
using TexLayer = VoxelBlockLayer<TexVoxel>;
using MeshLayer = BlockLayer<MeshBlock>;
using MeshUVLayer = BlockLayer<MeshBlockUV>;

}  // namespace nvblox

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

#include <nvblox/core/layer.h>
#include <nvblox/core/types.h>
#include <nvblox/mesh/mesh_block.h>

namespace nvblox {

// A structure which holds a Mesh.
// Generally produced as the result of fusing MeshBlocks in a Layer<MeshBlock>
// into a single mesh.
// NOTE(alexmillane): Currently only on the CPU.
struct Mesh {
  // Data
  std::vector<Vector3f> vertices;
  std::vector<Vector3f> normals;
  std::vector<int> triangles;
  std::vector<Color> colors;

  // Factory
  static Mesh fromLayer(const BlockLayer<MeshBlock>& layer);
};

// Same as a regular mesh but also defines uv coordinates for each vertex
struct MeshUV : Mesh {
  // Data
  std::vector<Vector2f> uvs;

  // Factory
  static MeshUV fromLayer(const BlockLayer<MeshBlockUV>& layer);
};

}  // namespace nvblox

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
#include "nvblox/mesh/mesh_block.h"

namespace nvblox {

MeshBlock::MeshBlock(MemoryType memory_type)
    : vertices(memory_type),
      normals(memory_type),
      triangles(memory_type),
      colors(memory_type),
      memory_type_(memory_type) {}

void MeshBlock::clear() {
  vertices.resize(0);
  normals.resize(0);
  triangles.resize(0);
  colors.resize(0);
  voxels.resize(0);
}

void MeshBlock::resizeToNumberOfVertices(size_t new_size) {
  vertices.resize(new_size);
  normals.resize(new_size);
  triangles.resize(new_size);
  voxels.resize(new_size);
}

void MeshBlock::reserveNumberOfVertices(size_t new_capacity) {
  vertices.reserve(new_capacity);
  normals.reserve(new_capacity);
  triangles.reserve(new_capacity);
  voxels.reserve(new_capacity);
}

MeshBlock::Ptr MeshBlock::allocate(MemoryType memory_type) {
  return std::make_shared<MeshBlock>(memory_type);
}

std::vector<Vector3f> MeshBlock::getVertexVectorOnCPU() const {
  return vertices.toVector();
}

std::vector<Vector3f> MeshBlock::getNormalVectorOnCPU() const {
  return normals.toVector();
}

std::vector<int> MeshBlock::getTriangleVectorOnCPU() const {
  return triangles.toVector();
}

std::vector<Color> MeshBlock::getColorVectorOnCPU() const {
  return colors.toVector();
}

size_t MeshBlock::size() const { return vertices.size(); }

size_t MeshBlock::capacity() const { return vertices.capacity(); }

void MeshBlock::expandColorsToMatchVertices() {
  colors.reserve(vertices.capacity());
  colors.resize(vertices.size());
}

void MeshBlock::expandIntensitiesToMatchVertices() {
  intensities.reserve(vertices.capacity());
  intensities.resize(vertices.size());
}

MeshBlockUV::MeshBlockUV(MemoryType memory_type)
    : MeshBlock(memory_type),
      uvs(memory_type),
      patches(memory_type),
      vertex_patches(memory_type) {}

void MeshBlockUV::clear() {
  MeshBlock::clear();
  uvs.clear();
  patches.clear();
  vertex_patches.clear();
}

void MeshBlockUV::expandUVsToMatchVertices() {
  uvs.reserve(vertices.capacity());
  uvs.resize(vertices.size());
}

void MeshBlockUV::expandVertexPatchesToMatchVertices() {
  vertex_patches.reserve(vertices.capacity());
  vertex_patches.resize(vertices.size());
}

std::vector<Vector2f> MeshBlockUV::getUVVectorOnCPU() const {
  return uvs.toVector();
}

std::vector<Color*> MeshBlockUV::getPatchVectorOnCPU() const {
  return patches.toVector();
}

std::vector<int> MeshBlockUV::getVertexPatchVectorOnCPU() const {
  return vertex_patches.toVector();
}

/**
 * @brief Adds a patch belonging to the index at (block_index, voxel_index) to
 * the vector of known patches if the corresponding index pair is new and
 * returns the patches' index. Otherwise no new patch is added and the index of
 * the known patch is returned.
 *
 * @param block_index
 * @param voxel_index
 * @param patch
 * @return int index of the patch in the patches vector
 */
int MeshBlockUV::addPatch(const Index3D& block_index,
                          const Index3D& voxel_index, const int rows,
                          const int cols, Color* patch) {
  // TODO(rasaford) because of the usage of std::find this function can
  // currently not be exectued on the GPU. --> Fix
  int patch_index = patches.size();
  // clang-format off
  const Index6D combined_index = Index6D( block_index(0), 
                                          block_index(1), 
                                          block_index(2), 
                                          voxel_index(0),
                                          voxel_index(1), 
                                          voxel_index(2));
  // clang-format on
  auto it = std::find(known_patch_indices.begin(), known_patch_indices.end(),
                      combined_index);

  if (it != known_patch_indices.end()) {
    patch_index = std::distance(known_patch_indices.begin(), it);
  } else {
    known_patch_indices.push_back(combined_index);
    patches.push_back(patch);
  }
  return patch_index;
}

MeshBlockUV::Ptr MeshBlockUV::allocate(MemoryType memory_type) {
  return std::make_shared<MeshBlockUV>(memory_type);
}

// Set the pointers to point to the mesh block.
CudaMeshBlock::CudaMeshBlock(MeshBlock* block) {
  CHECK_NOTNULL(block);
  vertices = block->vertices.data();
  normals = block->normals.data();
  triangles = block->triangles.data();
  colors = block->colors.data();
  voxels = block->voxels.data();
  size = block->vertices.size();
}

CudaMeshBlockUV::CudaMeshBlockUV(MeshBlockUV* block) : CudaMeshBlock(block) {
  uvs_ = block->uvs.data();
}

}  // namespace nvblox
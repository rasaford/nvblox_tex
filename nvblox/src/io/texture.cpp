#include "nvblox/io/texture.h"
#include <cmath>
#include <map>
#include <memory>

namespace nvblox {
namespace io {

// std::unique_ptr<cv::Mat> packTextures(MeshUVLayer& mesh_layer,
//                                       const MemoryType memory_type,
//                                       const int padding) {
//   return packTextures(mesh_layer, mesh_layer.getAllBlockIndices(),
//   memory_type,
//                       padding);
// }

std::unique_ptr<TexturedMesh> packTextures(const MeshUVLayer& mesh_layer,
                                           const int padding) {
  return packTextures(mesh_layer, mesh_layer.getAllBlockIndices(), padding);
}

std::unique_ptr<TexturedMesh> packTextures(
    const MeshUVLayer& mesh_layer, const std::vector<Index3D> block_indices,
    const int padding) {
  const int patch_width = TexVoxel::kPatchWidth;
  MeshUV mesh = MeshUV::fromLayer(mesh_layer, block_indices);

  // collect all patches and corresponding patch_vertices
  std::vector<Color*> patches;
  std::unordered_map<int, std::vector<int>> patch_vertices;
  int num_vertices = 0;

  for (const Index3D& block_index : block_indices) {
    MeshBlockUV::ConstPtr block_ptr = mesh_layer.getBlockAtIndex(block_index);
    if (block_ptr == nullptr) {
      continue;
    }
    const std::vector<Color*> block_patches = block_ptr->getPatchVectorOnCPU();
    // patches.reserve(patches.end() + block_ptr->patches.size());
    patches.insert(patches.end(), block_ptr->patches.begin(),
                   block_ptr->patches.end());

    // insert the vertices in the mesh every patch belongs to
    // NOTE(rasaford) this assumes that the vertices in MeshUV::fromLayer() are
    // inserted sequentially and in the order given by block_indices
    const std::vector<int> block_vertex_patches =
        block_ptr->getVertexPatchVectorOnCPU();
    for (int i = 0; i < block_vertex_patches.size(); ++i) {
      const int patch_id = block_vertex_patches[i];
      patch_vertices[patch_id].push_back(num_vertices + i);
    }
    num_vertices += block_vertex_patches.size();
  }
  CHECK_EQ(mesh.vertices.size(), num_vertices);

  const int texture_patch_width = static_cast<int>(ceil(sqrt(patches.size())));
  const int padded_patch_width = patch_width + padding;
  const int texture_width = padded_patch_width * texture_patch_width;

  // allocate a texture for all patches (with padding) to fit into
  cv::Mat texture =
      cv::Mat::zeros(cv::Size(texture_width, texture_width), CV_8UC3);
  cv::Mat patch_colors =
      cv::Mat::zeros(cv::Size(patch_width, patch_width), CV_8UC3);
  std::vector<uint8_t> patch_bytes;
  patch_bytes.reserve(3 * patch_width * patch_width);

  // Offset UV coordinates in mesh and insert the corresponding patch in each
  // texture tile
  for (int patch_idx = 0; patch_idx < patches.size(); ++patch_idx) {
    const Color* patch = patches[patch_idx];
    const Index2D top_left(
        padded_patch_width * (patch_idx % texture_patch_width),
        padded_patch_width * (patch_idx / texture_patch_width));
    const Vector2f top_left_uv = top_left.cast<float>() / texture_width;

    // Offset and scale patch uvs to fit in the global texture
    for (const int& vertex_idx : patch_vertices[patch_idx]) {
      const Vector2f offset_uvs =
          top_left_uv + (patch_width / static_cast<float>(texture_width)) *
                            mesh.uvs[vertex_idx];
      mesh.uvs[vertex_idx] = offset_uvs;
    }

    // copy the custom Color objects to the global texture buffer
    patch_bytes.clear();
    for (size_t i = 0; i < patch_width * patch_width; ++i) {
      // opencv matrices are BGR unless defined otherwise
      const int x = i % patch_width, y = i / patch_width;
      patch_colors.at<cv::Vec3b>(x, y) =
          cv::Vec3b(patch[i].b, patch[i].g, patch[i].r);
    }
    patch_colors.copyTo(texture(cv::Rect(
        cv::Point(top_left[0], top_left[1]),
        cv::Point(top_left[0] + patch_width, top_left[1] + patch_width))));
  }

  // allocate heap memory for the object and it's contents
  return std::make_unique<TexturedMesh>(texture, mesh);
}

}  // namespace io
}  // namespace nvblox
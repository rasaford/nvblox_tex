#include "nvblox/io/texture.h"
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>

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
                                           const TexLayer& tex_layer,
                                           const int padding) {
  return packTextures(mesh_layer, tex_layer, mesh_layer.getAllBlockIndices(),
                      padding);
}

std::unique_ptr<TexturedMesh> packTextures(
    const MeshUVLayer& mesh_layer, const TexLayer& tex_layer,
    const std::vector<Index3D>& block_indices, const int padding) {
  const int patch_width = TexVoxel::kPatchWidth;
  MeshUV mesh = MeshUV::fromLayer(mesh_layer, block_indices);

  // vector of all color patches to pack
  std::vector<Color*> patches;
  // for each index in the patches vector this map gives the corresponding
  // vertex indices in the merged mesh
  std::unordered_map<int, std::vector<int>> patch_vertices;
  int num_vertices = 0;
  int num_patches = 0;
  for (const Index3D& block_index : block_indices) {
    MeshBlockUV::ConstPtr mesh_block = mesh_layer.getBlockAtIndex(block_index);
    TexBlock::ConstPtr tex_block = tex_layer.getBlockAtIndex(block_index);
    if (tex_block.get() != nullptr) {
      constexpr size_t kVoxelsPerSide3 = TexBlock::kVoxelsPerSide *
                                         TexBlock::kVoxelsPerSide *
                                         TexBlock::kVoxelsPerSide;

      patches.resize(num_patches + kVoxelsPerSide3);
      std::vector<Index3D> voxels = mesh_block->getVoxelsVectorOnCPU();

      for (int i = 0; i < voxels.size(); i++) {
        const Index3D& v = voxels[i];
        const int linear_idx =
            TexBlock::kVoxelsPerSide * TexBlock::kVoxelsPerSide * v[0] +
            TexBlock::kVoxelsPerSide * v[1] + v[2];
        const int patch_idx = num_patches + linear_idx;
        const int vertex_idx = num_vertices + i;
        patches[patch_idx] =
            const_cast<Color*>(tex_block->voxels[v[0]][v[1]][v[2]].colors);
        patch_vertices[patch_idx].push_back(vertex_idx);
      }
      num_patches += kVoxelsPerSide3;
    }
    num_vertices += mesh_block->vertices.size();
  }
  // Sanity checks to make sure block level indices have been correctly
  // converted to global indices
  CHECK_EQ(mesh.vertices.size(), num_vertices);
  CHECK_EQ(patches.size(), num_patches);

  const int num_valid_patches = std::accumulate(
      patches.begin(), patches.end(), 0,
      [](int a, Color* ptr) { return a + static_cast<int>(ptr != nullptr); });

  const int texture_patch_width =
      static_cast<int>(ceil(sqrt(num_valid_patches)));
  const int padded_patch_width = patch_width + 2 * padding;
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
  int patch_cnt = 0;
  for (int patch_idx = 0; patch_idx < patches.size(); ++patch_idx) {
    const Color* patch = patches[patch_idx];
    if (patch == nullptr) {
      continue;
    }
    const Index2D top_left(
        1 + padded_patch_width * (patch_cnt % texture_patch_width),
        1 + padded_patch_width * (patch_cnt / texture_patch_width));
    const Vector2f top_left_uv = top_left.array().cast<float>() / texture_width;

    // Offset and scale patch uvs to fit in the global texture
    for (const int& vertex_idx : patch_vertices[patch_idx]) {
      Vector2f offset_uvs =
          top_left_uv + (patch_width / static_cast<float>(texture_width)) *
                            mesh.uvs[vertex_idx];
      // flip y axis for proper display in blender
      offset_uvs(1) = 1 - offset_uvs(1);

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
    // copy scaled up version of each patch around the actual patch, to fill the
    // padding space around it with color value that are similar to the valid
    // patch ones. This prevents black borders around triangles at the borders
    // of a patch when using some kind of texture interpolation in e.g. blender
    const Index2D bottom_right = top_left + Index2D(patch_width, patch_width);
    const Index2D top_left_padded =
        (top_left - Index2D(1, 1)).cwiseMax(Index2D::Zero());
    const Index2D bottom_right_padded =
        bottom_right +
        Index2D(1, 1).cwiseMin(Index2D(texture_width, texture_width));

    cv::Mat target_patch_padded = texture(
        cv::Rect(cv::Point(top_left_padded[0], top_left_padded[1]),
                 cv::Point(bottom_right_padded[0], bottom_right_padded[1])));

    cv::resize(patch_colors, target_patch_padded, target_patch_padded.size(), 0,
               0, cv::INTER_LINEAR);
    patch_colors.copyTo(texture(cv::Rect(
        cv::Point(top_left[0], top_left[1]),
        cv::Point(top_left[0] + patch_width, top_left[1] + patch_width))));
    patch_cnt++;
  }

  // allocate heap memory for the object and it's contents
  return std::make_unique<TexturedMesh>(texture, mesh);
}

}  // namespace io
}  // namespace nvblox
#pragma once
#include <opencv2/opencv.hpp>
#include "nvblox/core/color.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/image.h"
#include "nvblox/core/types.h"
#include "nvblox/mesh/mesh.h"
#include "nvblox/mesh/mesh_block.h"

namespace nvblox {
namespace io {

struct TexturedMesh {
  // texture of the entire mesh defined as a cv::Mat to be able to use opencv's
  // file io functionality
  cv::Mat texture;
  // TODO(rasaford) add the mesh's material here as well
  MeshUV mesh;

  TexturedMesh(cv::Mat texture, MeshUV mesh)
      : texture(std::move(texture)), mesh(std::move(mesh)) {}
};

/**
 * @brief Packs the texture patches of all voxles in all blocks into a single
 * texture canvas.
 *
 * @param mesh_layer
 * @param padding padding around each texture patch, defined in pixels
 * @return std::unique_ptr<TexturedMesh>
 */
std::unique_ptr<TexturedMesh> packTextures(const MeshUVLayer& mesh_layer, const TexLayer& tex_layer,
                                           const int padding = 1);

/**
 * @brief Packs the texture patches of all voxles in the blocks given by
 * block_indices into a single texture canvas.
 *
 * @param mesh_layer
 * @param block_indices
 * @param padding padding around each texture patch, defined in pixels
 * @return std::unique_ptr<TexturedMesh>
 */
std::unique_ptr<TexturedMesh> packTextures(
    const MeshUVLayer& mesh_layer, const TexLayer& tex_layer, const std::vector<Index3D>& block_indices,
    const int padding = 1);

}  // namespace io
}  // namespace nvblox
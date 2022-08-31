#include <cuda_runtime.h>

#include <utility>
#include "nvblox/core/accessors.h"
#include "nvblox/core/common_names.h"
#include "nvblox/integrators/integrators_common.h"
#include "nvblox/mesh/impl/marching_cubes_table.h"
#include "nvblox/mesh/marching_cubes.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

__host__ __device__ inline bool projectToUV(const Vector3f& vertex,
                                            const Vector3f& voxel_center,
                                            const float voxel_size,
                                            const TexVoxel::Dir direction,
                                            Vector2f* uv) {
  // NOTE(rasaford) since the directions encoded in TexVoxel::Dir are aligned
  // with the major coordinate axes, we do not need to do a complicated
  // projection here but can just take the respective coordinates directly
  Vector3f texel_coords = (vertex - voxel_center) / voxel_size;
  texel_coords =
      texel_coords.cwiseMax(Vector3f::Zero()).cwiseMin(Vector3f::Ones());
  if (direction == TexVoxel::Dir::X_PLUS ||
      direction == TexVoxel::Dir::X_MINUS) {
    *uv = Vector2f(texel_coords(1), texel_coords(2));
    return true;
  } else if (direction == TexVoxel::Dir::Y_PLUS ||
             direction == TexVoxel::Dir::Y_MINUS) {
    *uv = Vector2f(texel_coords(0), texel_coords(2));
    return true;
  } else if (direction == TexVoxel::Dir::Z_PLUS ||
             direction == TexVoxel::Dir::Z_MINUS) {
    *uv = Vector2f(texel_coords(0), texel_coords(1));
    return true;
  } else {
    return false;
  }
}
__host__ __device__ inline Color blendTwoColors(const Color& first_color,
                                                float first_weight,
                                                const Color& second_color,
                                                float second_weight) {
  float total_weight = first_weight + second_weight;

  first_weight /= total_weight;
  second_weight /= total_weight;

  Color new_color;
  new_color.r = static_cast<uint8_t>(std::round(
      first_color.r * first_weight + second_color.r * second_weight));
  new_color.g = static_cast<uint8_t>(std::round(
      first_color.g * first_weight + second_color.g * second_weight));
  new_color.b = static_cast<uint8_t>(std::round(
      first_color.b * first_weight + second_color.b * second_weight));

  return new_color;
}

__host__ __device__ inline Color getDirColor(
    const TexVoxel::Dir dir, const float positive_weight = 0.5f) {
  Color color;
  switch (dir) {
    case TexVoxel::Dir::X_PLUS:
      return Color::Red();
    case TexVoxel::Dir::X_MINUS:
      return blendTwoColors(Color::Red(), positive_weight, Color::Black(),
                            1 - positive_weight);
    case TexVoxel::Dir::Y_PLUS:
      return Color::Green();
    case TexVoxel::Dir::Y_MINUS:
      return blendTwoColors(Color::Green(), positive_weight, Color::Black(),
                            1 - positive_weight);
    case TexVoxel::Dir::Z_PLUS:
      return Color::Blue();
    case TexVoxel::Dir::Z_MINUS:
      return blendTwoColors(Color::Blue(), positive_weight, Color::Black(),
                            1 - positive_weight);
    default:
      return Color::Yellow();
  }
}

void MeshUVIntegrator::textureMesh(const TexLayer& tex_layer,
                                   MeshUVLayer* mesh_layer) {
  textureMesh(tex_layer, mesh_layer->getAllBlockIndices(), mesh_layer);
}
void MeshUVIntegrator::textureMesh(const TexLayer& tex_layer,
                                   const std::vector<Index3D>& block_indices,
                                   MeshUVLayer* mesh_layer) {
  // Default choice is GPU
  textureMeshGPU(tex_layer, block_indices, mesh_layer);
}

__global__ void colorMeshBlockByClosestTexVoxel(const TexBlock** tex_blocks,
                                                const Index3D* block_indices,
                                                const float block_size,
                                                const float voxel_size,
                                                CudaMeshBlockUV* mesh_blocks) {
  const Index3D block_idx = block_indices[blockIdx.x];
  const TexBlock* tex_block = tex_blocks[blockIdx.x];
  CudaMeshBlockUV mesh_block = mesh_blocks[blockIdx.x];
  if (tex_block == nullptr) {
    return;
  }

  // grid stride access pattern since this memory is already on the GPU
  for (int v_idx = threadIdx.x; v_idx < mesh_block.size; v_idx += blockDim.x) {
    const Vector3f vertex = mesh_block.vertices[v_idx];
    const Index3D voxel_idx = mesh_block.voxels[v_idx];
    const TexVoxel tex_voxel =
        tex_block->voxels[voxel_idx[0]][voxel_idx[1]][voxel_idx[2]];

    Vector3f voxel_center;
    voxel_center = getCenterPostionFromBlockIndexAndVoxelIndex(
        block_size, block_idx, voxel_idx);

    Vector2f patch_uv;
    if (projectToUV(vertex, voxel_center, voxel_size, tex_voxel.dir,
                    &patch_uv)) {
      // update the block attributes in global memory
      mesh_block.colors[v_idx] = getDirColor(tex_voxel.dir);
      mesh_block.uvs[v_idx] = patch_uv;
    } else {
      mesh_block.colors[v_idx] = Color::Pink();
      mesh_block.uvs[v_idx] = Vector2f::Zero();
    }
  }
}

void MeshUVIntegrator::textureMeshGPU(const TexLayer& tex_layer,
                                      MeshUVLayer* mesh_layer) {
  textureMeshGPU(tex_layer, mesh_layer->getAllBlockIndices(), mesh_layer);
}

void MeshUVIntegrator::textureMeshGPU(
    const TexLayer& tex_layer,
    const std::vector<Index3D>& requested_block_indices,
    MeshUVLayer* mesh_layer) {
  CHECK_NOTNULL(mesh_layer);
  CHECK_EQ(tex_layer.block_size(), mesh_layer->block_size());

  timing::Timer("mesh/gpu/texture");

  // NOTE(alexmillane): Generally, some of the MeshBlocks which we are
  // "coloring" will not have data in the color layer. HOWEVER, for colored
  // MeshBlocks (ie with non-empty color members), the size of the colors must
  // match vertices. Therefore we "color" all requested block_indices in two
  // parts:
  // - The first part using the color layer, and
  // - the second part a constant color.

  // allocate space for uvs
  std::vector<Index3D> block_indices;
  block_indices.reserve(requested_block_indices.size());
  std::vector<MeshBlockUV*> blocks;
  blocks.reserve(requested_block_indices.size());

  for (const Index3D& block_idx : requested_block_indices) {
    typename MeshBlockUV::Ptr block = mesh_layer->getBlockAtIndex(block_idx);
    if (block == nullptr) {
      continue;
    }
    // pre-allocate all buffers to the required size
    block->expandColorsToMatchVertices();
    block->expandUVsToMatchVertices();
    block_indices.push_back(block_idx);
    blocks.push_back(block.get());
  }

  const size_t num_blocks = block_indices.size();
  if (num_blocks == 0) {
    return;
  }

  // expand buffers
  if (num_blocks > block_indices_device_.size()) {
    const size_t new_size = static_cast<size_t>(1.5 * num_blocks);
    block_indices_device_.reserve(new_size);
    tex_blocks_host_.reserve(new_size);
    tex_blocks_device_.reserve(new_size);
    uv_mesh_blocks_host_.reserve(new_size);
    uv_mesh_blocks_device_.reserve(new_size);
  }

  // NOTE(rasaford): Overwrite the vector elements, instead of push_back to
  // avoid unnecessary allocations (since clear free's the vector's memory).
  // Resize to set the vector's size to blocks.size(). Since we reserved memory
  // before, this will not trigger an internal vector reserve.
  tex_blocks_host_.resize(blocks.size());
  uv_mesh_blocks_host_.resize(blocks.size());
  for (size_t i = 0; i < blocks.size(); i++) {
    MeshBlockUV* block = blocks[i];
    typename TexBlock::ConstPtr tex_block =
        tex_layer.getBlockAtIndex(block_indices[i]);
    tex_blocks_host_[i] = tex_block.get();
    uv_mesh_blocks_host_[i] = CudaMeshBlockUV(block);
  }

  // copy host -> device (vector sizes are adjusted automatically)
  block_indices_device_ = std::move(block_indices);
  tex_blocks_device_ = tex_blocks_host_;
  uv_mesh_blocks_device_ = uv_mesh_blocks_host_;

  const dim3 kThreadsPerBlock(32, 1, 1);
  // clang-format off
  colorMeshBlockByClosestTexVoxel<<<num_blocks, kThreadsPerBlock, 0, cuda_stream_>>>(
                                              tex_blocks_device_.data(), 
                                              block_indices_device_.data(),
                                              tex_layer.block_size(), 
                                              tex_layer.voxel_size(),
                                              uv_mesh_blocks_device_.data());
  // clang-format on
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());
}

void MeshUVIntegrator::textureMeshCPU(const TexLayer& tex_layer,
                                      MeshUVLayer* mesh_layer) {
  textureMeshCPU(tex_layer, mesh_layer->getAllBlockIndices(), mesh_layer);
}

void MeshUVIntegrator::textureMeshCPU(const TexLayer& tex_layer,
                                      const std::vector<Index3D>& block_indices,
                                      MeshUVLayer* mesh_layer) {
  timing::Timer("mesh/cpu/texture");
  // For each vertex just grab the closest color
  for (const Index3D& block_idx : block_indices) {
    MeshBlockUV::Ptr block = mesh_layer->getBlockAtIndex(block_idx);
    if (block == nullptr) {
      continue;
    }
    // pre-allocate all buffers to the required size
    block->expandColorsToMatchVertices();
    block->expandUVsToMatchVertices();

    for (int i = 0; i < block->vertices.size(); i++) {
      const Vector3f& vertex = block->vertices[i];
      const Index3D& voxel_idx = block->voxels[i];
      const TexVoxel* tex_voxel =
          getVoxelAtBlockAndVoxelIndex(tex_layer, block_idx, voxel_idx);
      bool colors_updated = false;

      // not all blocks in a layer might be allocated. So we check if a voxel
      // exists at the vertex position.
      if (tex_voxel != nullptr) {
        const Vector3f voxel_center =
            getCenterPostionFromBlockIndexAndVoxelIndex(tex_layer.block_size(),
                                                        block_idx, voxel_idx);

        Vector2f patch_uv;
        if (projectToUV(vertex, voxel_center, tex_layer.voxel_size(),
                        tex_voxel->dir, &patch_uv)) {
          colors_updated = true;
          block->colors[i] = getDirColor(tex_voxel->dir);
          block->uvs[i] = patch_uv;
        }
      }
      if (!colors_updated) {
        block->colors[i] = Color::Gray();
        block->uvs[i] = Vector2f::Zero();
      }
    }
  }
}

}  // namespace nvblox
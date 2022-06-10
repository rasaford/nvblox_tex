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
  switch (direction) {
    case TexVoxel::Dir::X_PLUS:
    case TexVoxel::Dir::X_MINUS:
      *uv << texel_coords(1), texel_coords(2);
      return true;
    case TexVoxel::Dir::Y_PLUS:
    case TexVoxel::Dir::Y_MINUS:
      *uv << texel_coords(0), texel_coords(2);
      return true;
    case TexVoxel::Dir::Z_PLUS:
    case TexVoxel::Dir::Z_MINUS:
      *uv << texel_coords(0), texel_coords(1);
      return true;
    default:
      return false;
  }
}

__host__ __device__ inline Color getDirColor(
    const TexVoxel::Dir dir, const float positive_weight = 0.5f) {
  Color color;
  switch (dir) {
    case TexVoxel::Dir::X_PLUS:
      color = Color::Red();
      break;
    case TexVoxel::Dir::X_MINUS:
      color = Color::blendTwoColors(Color::Red(), positive_weight,
                                    Color::Black(), 1 - positive_weight);
      break;
    case TexVoxel::Dir::Y_PLUS:
      color = Color::Green();
      break;
    case TexVoxel::Dir::Y_MINUS:
      color = Color::blendTwoColors(Color::Green(), positive_weight,
                                    Color::Black(), 1 - positive_weight);
      break;
    case TexVoxel::Dir::Z_PLUS:
      color = Color::Blue();
      break;
    case TexVoxel::Dir::Z_MINUS:
      color = Color::blendTwoColors(Color::Blue(), positive_weight,
                                    Color::Black(), 1 - positive_weight);
      break;
    default:
      color = Color::Gray();
      break;
  }
  return color;
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

void MeshUVIntegrator::textureMeshGPU(const TexLayer& tex_layer,
                                      MeshUVLayer* mesh_layer) {
  textureMeshGPU(tex_layer, mesh_layer->getAllBlockIndices(), mesh_layer);
}

__global__ void fillMeshBlocksGPU(const Index3D* block_indices,
                                  const TexBlock** tex_blocks,
                                  const float block_size,
                                  const float voxel_size,
                                  CudaMeshBlockUV* mesh_blocks) {
  const Index3D block_idx = block_indices[blockIdx.x];
  const TexBlock* tex_block = tex_blocks[blockIdx.x];
  CudaMeshBlockUV mesh_block = mesh_blocks[blockIdx.x];
  Vector2f patch_uv;

  // Iterate though CudaMeshBlockUV vertices -- strided access pattern
  for (int i = threadIdx.x; i < mesh_block.size; i += blockDim.x) {
    const Vector3f vertex = mesh_block.vertices[i];
    const Index3D voxel_idx = mesh_block.voxels[i];
    const TexVoxel* tex_voxel =
        &tex_block->voxels[voxel_idx.x()][voxel_idx.y()][voxel_idx.z()];

    const Vector3f voxel_center = getCenterPostionFromBlockIndexAndVoxelIndex(
        block_size, block_idx, voxel_idx);

    const int linear_idx =
        linearizeVoxelIndex(voxel_idx, TexBlock::kVoxelsPerSide);

    // potentially many threads might write to a given linear_idx in the patches
    // array. Therefore, assigning the associated patch to it needs to be
    // atomic. This is done with the buildint CompareAndSwap (CAS) function.
    // Since it natively does not operate on pointers, we cast them explicitly
    // to the Address type defined below. This is equivalent to:
    // mesh_block.patches[linear_idx] =
    //      mesh_block.patches[linear_idx] == nullptr ?
    //            tex_voxel->colors : mesh_block.patches[linear_idx]
    using Address = unsigned long long int;
    atomicCAS((Address*)&mesh_block.patches[linear_idx], (Address) nullptr,
              (Address) const_cast<TexVoxel*>(tex_voxel)->colors);
    // mesh_block.patches[linear_idx] = const_cast<TexVoxel*>(tex_voxel)->colors;

    if (projectToUV(vertex, voxel_center, voxel_size, tex_voxel->dir,
                    &patch_uv)) {
      mesh_block.colors[i] = getDirColor(tex_voxel->dir);
      mesh_block.uvs[i] = patch_uv;
      mesh_block.vertex_patches[i] =
          linearizeVoxelIndex(voxel_idx, TexBlock::kVoxelsPerSide);
    } else {
      mesh_block.colors[i] = Color::Gray();
      mesh_block.uvs[i] = Vector2f::Zero();
      mesh_block.vertex_patches[i] = -1;
    }
  }
}

void MeshUVIntegrator::projectTexToMeshGPU(
    const TexLayer& tex_layer, const std::vector<Index3D> block_indices,
    MeshUVLayer* mesh_layer) {
  CHECK_NOTNULL(mesh_layer);
  if (block_indices.size() == 0) {
    return;
  }

  host_vector<const TexBlock*> tex_blocks;
  tex_blocks.reserve(block_indices.size());
  for (const auto& block_ptr :
       getBlockPtrsFromIndices<TexBlock>(block_indices, tex_layer)) {
    tex_blocks.push_back(block_ptr);
  }

  host_vector<CudaMeshBlockUV> cuda_mesh_blocks;
  cuda_mesh_blocks.reserve(block_indices.size());
  for (const auto& block_idx : block_indices) {
    cuda_mesh_blocks.push_back(
        CudaMeshBlockUV(mesh_layer->getBlockAtIndex(block_idx).get()));
  }

  device_vector<Index3D> device_block_indices = block_indices;
  device_vector<const TexBlock*> device_tex_blocks = tex_blocks;
  device_vector<CudaMeshBlockUV> device_cuda_mesh_blocks = cuda_mesh_blocks;

  const int num_blocks = block_indices.size();
  const int kThreadsPerBlock = 8 * 32;  // Chosen at random
  // clang-format off
  fillMeshBlocksGPU<<<num_blocks, kThreadsPerBlock, 0, cuda_stream_>>>(
      device_block_indices.data(), 
      device_tex_blocks.data(),
      tex_layer.block_size(), 
      tex_layer.voxel_size(),
      device_cuda_mesh_blocks.data());
  // clang-format on
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());
}

void MeshUVIntegrator::textureMeshGPU(
    const TexLayer& tex_layer,
    const std::vector<Index3D>& requested_block_indices,
    MeshUVLayer* mesh_layer) {
  // For each requested index check if it exists then allocate space in all
  // required buffers. We only work on existing indices
  std::vector<Index3D> block_indices;
  block_indices.reserve(requested_block_indices.size());
  for (const Index3D& block_idx : requested_block_indices) {
    MeshBlockUV::Ptr mesh_block = mesh_layer->getBlockAtIndex(block_idx);
    TexBlock::ConstPtr tex_block = tex_layer.getBlockAtIndex(block_idx);
    if (mesh_block == nullptr || tex_block == nullptr) {
      continue;
    }
    // pre-allocate all buffers to the required size
    mesh_block->expandColorsToMatchVertices();
    mesh_block->expandUVsToMatchVertices();
    mesh_block->expandVertexPatchesToMatchVertices();
    mesh_block->expandPatchesToMatchNumVoxels(TexBlock::kVoxelsPerSide);
    block_indices.push_back(block_idx);
  }

  projectTexToMeshGPU(tex_layer, block_indices, mesh_layer);
}

void MeshUVIntegrator::textureMeshCPU(const TexLayer& tex_layer,
                                      MeshUVLayer* mesh_layer) {
  textureMeshCPU(tex_layer, mesh_layer->getAllBlockIndices(), mesh_layer);
}

void MeshUVIntegrator::textureMeshCPU(const TexLayer& tex_layer,
                                      const std::vector<Index3D>& block_indices,
                                      MeshUVLayer* mesh_layer) {
  // For each vertex just grab the closest color
  for (const Index3D& block_idx : block_indices) {
    MeshBlockUV::Ptr block = mesh_layer->getBlockAtIndex(block_idx);
    if (block == nullptr) {
      continue;
    }
    // pre-allocate all buffers to the required size
    block->expandColorsToMatchVertices();
    block->expandUVsToMatchVertices();
    block->expandVertexPatchesToMatchVertices();

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

        // Add the tex_voxel's colors as a patch to the MeshBlockUV.
        // NOTE(rasaford) Since getVoxelAtPosition() requires a const pointer,
        // we have to do this ugly const_cast here. In reality the
        // TexVoxel*->colors attribute is non-const anyways.
        const int patch_index = block->addPatch(
            block_idx, voxel_idx, tex_voxel->kPatchWidth,
            tex_voxel->kPatchWidth, const_cast<TexVoxel*>(tex_voxel)->colors);

        Vector2f patch_uv;
        if (projectToUV(vertex, voxel_center, tex_layer.voxel_size(),
                        tex_voxel->dir, &patch_uv)) {
          colors_updated = true;
          block->colors[i] = getDirColor(tex_voxel->dir);
          block->uvs[i] = patch_uv;
          block->vertex_patches[i] = patch_index;
        }
      }
      if (!colors_updated) {
        block->colors[i] = default_mesh_color_;
        block->uvs[i] = Vector2f::Zero();
      }
    }
  }
}

}  // namespace nvblox
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

//
/* Texture Mesh blocks on the GPU
 *
 * Call with
 * - one ThreadBlock per VoxelBlock, GridDim 1D
 * - BlockDim 1D, any size: we implement a stridded access pattern over
 *   MeshBlock verticies
 *
 * @param: color_blocks:     a list of color blocks which correspond in position
 *                           to mesh_blocks
 * @param: block_indices:    a list of blocks indices.
 * @param: cuda_mesh_blocks: a list of mesh_blocks to be colored.
 */
// __global__ void colorMeshBlockByClosestColorVoxel(
//     const ColorBlock** color_blocks, const Index3D* block_indices,
//     const float block_size, const float voxel_size,
//     CudaMeshBlock* cuda_mesh_blocks) {
//   // Block
//   const ColorBlock* color_block_ptr = color_blocks[blockIdx.x];
//   const Index3D block_index = block_indices[blockIdx.x];
//   CudaMeshBlock cuda_mesh_block = cuda_mesh_blocks[blockIdx.x];

//   // The position of this block in the layer
//   const Vector3f p_L_B_m = getPositionFromBlockIndex(block_size,
//   block_index);

//   // Interate through MeshBlock vertices - Stidded access pattern
//   for (int i = threadIdx.x; i < cuda_mesh_block.size; i += blockDim.x) {
//     // The position of this vertex in the layer
//     const Vector3f p_L_V_m = cuda_mesh_block.vertices[i];

//     // The position of this vertex in the block
//     const Vector3f p_B_V_m = p_L_V_m - p_L_B_m;

//     // Convert this to a voxel index
//     Index3D voxel_idx_in_block = (p_B_V_m.array() / voxel_size).cast<int>();

//     // NOTE(alexmillane): Here we make some assumptions.
//     // - We assume that the closest voxel to p_L_V is in the ColorBlock
//     //   co-located with the MeshBlock from which p_L_V was drawn.
//     // - This is will (very?) occationally be incorrect when mesh vertices
//     //   escape block boundaries. However, making this assumption saves us
//     any
//     //   neighbour calculations.
//     constexpr size_t KVoxelsPerSizeMinusOne =
//         VoxelBlock<ColorVoxel>::kVoxelsPerSide - 1;
//     voxel_idx_in_block =
//         voxel_idx_in_block.array().min(KVoxelsPerSizeMinusOne).max(0);

//     // Get the color voxel
//     const ColorVoxel color_voxel =
//         color_block_ptr->voxels[voxel_idx_in_block.x()]  // NOLINT
//                                [voxel_idx_in_block.y()]  // NOLINT
//                                [voxel_idx_in_block.z()];

//     // Write the color out to global memory
//     cuda_mesh_block.colors[i] = color_voxel.color;
//   }
// }

// __global__ void colorMeshBlocksConstant(Color color,
//                                         CudaMeshBlock* cuda_mesh_blocks) {
//   // Each threadBlock operates on a single MeshBlock
//   CudaMeshBlock cuda_mesh_block = cuda_mesh_blocks[blockIdx.x];
//   // Interate through MeshBlock vertices - Stidded access pattern
//   for (int i = threadIdx.x; i < cuda_mesh_block.size; i += blockDim.x) {
//     cuda_mesh_block.colors[i] = color;
//   }
// }

// void colorMeshBlocksConstantGPU(const std::vector<Index3D>& block_indices,
//                                 const Color& color, MeshLayer* mesh_layer,
//                                 cudaStream_t cuda_stream) {
//   CHECK_NOTNULL(mesh_layer);
//   if (block_indices.size() == 0) {
//     return;
//   }

//   // Prepare CudaMeshBlocks, which are effectively containers of device
//   pointers std::vector<CudaMeshBlock> cuda_mesh_blocks;
//   cuda_mesh_blocks.resize(block_indices.size());
//   for (int i = 0; i < block_indices.size(); i++) {
//     cuda_mesh_blocks[i] =
//         CudaMeshBlock(mesh_layer->getBlockAtIndex(block_indices[i]).get());
//   }

//   // Allocate
//   CudaMeshBlock* cuda_mesh_block_device_ptrs;
//   checkCudaErrors(cudaMalloc(&cuda_mesh_block_device_ptrs,
//                              cuda_mesh_blocks.size() *
//                              sizeof(CudaMeshBlock)));

//   // Host -> GPU
//   checkCudaErrors(
//       cudaMemcpyAsync(cuda_mesh_block_device_ptrs, cuda_mesh_blocks.data(),
//                       cuda_mesh_blocks.size() * sizeof(CudaMeshBlock),
//                       cudaMemcpyHostToDevice, cuda_stream));

//   // Kernel call - One ThreadBlock launched per VoxelBlock
//   constexpr int kThreadsPerBlock = 8 * 32;  // Chosen at random
//   const int num_blocks = block_indices.size();
//   // colorMeshBlocksConstant<<<num_blocks, kThreadsPerBlock, 0,
//   cuda_stream>>>(
//   //     Color::Gray(),  // NOLINT
//   //     cuda_mesh_block_device_ptrs);
//   checkCudaErrors(cudaStreamSynchronize(cuda_stream));
//   checkCudaErrors(cudaPeekAtLastError());

//   // Deallocate
//   checkCudaErrors(cudaFree(cuda_mesh_block_device_ptrs));
// }

// void colorMeshBlockByClosestColorVoxelGPU(
//     const ColorLayer& color_layer, const std::vector<Index3D>& block_indices,
//     MeshLayer* mesh_layer, cudaStream_t cuda_stream) {
//   CHECK_NOTNULL(mesh_layer);
//   if (block_indices.size() == 0) {
//     return;
//   }

//   // Get the locations (on device) of the color blocks
//   // NOTE(alexmillane): This function assumes that all block_indices have
//   been
//   // checked to exist in color_layer.
//   std::vector<const ColorBlock*> color_blocks =
//       getBlockPtrsFromIndices(block_indices, color_layer);

//   // Prepare CudaMeshBlocks, which are effectively containers of device
//   pointers std::vector<CudaMeshBlock> cuda_mesh_blocks;
//   cuda_mesh_blocks.resize(block_indices.size());
//   for (int i = 0; i < block_indices.size(); i++) {
//     cuda_mesh_blocks[i] =
//         CudaMeshBlock(mesh_layer->getBlockAtIndex(block_indices[i]).get());
//   }

//   // Allocate
//   const ColorBlock** color_block_device_ptrs;
//   checkCudaErrors(cudaMalloc(&color_block_device_ptrs,
//                              color_blocks.size() * sizeof(ColorBlock*)));
//   Index3D* block_indices_device_ptr;
//   checkCudaErrors(cudaMalloc(&block_indices_device_ptr,
//                              block_indices.size() * sizeof(Index3D)));
//   CudaMeshBlock* cuda_mesh_block_device_ptrs;
//   checkCudaErrors(cudaMalloc(&cuda_mesh_block_device_ptrs,
//                              cuda_mesh_blocks.size() *
//                              sizeof(CudaMeshBlock)));

//   // Host -> GPU transfers
//   checkCudaErrors(cudaMemcpyAsync(color_block_device_ptrs,
//   color_blocks.data(),
//                                   color_blocks.size() * sizeof(ColorBlock*),
//                                   cudaMemcpyHostToDevice, cuda_stream));
//   checkCudaErrors(cudaMemcpyAsync(block_indices_device_ptr,
//                                   block_indices.data(),
//                                   block_indices.size() * sizeof(Index3D),
//                                   cudaMemcpyHostToDevice, cuda_stream));
//   checkCudaErrors(
//       cudaMemcpyAsync(cuda_mesh_block_device_ptrs, cuda_mesh_blocks.data(),
//                       cuda_mesh_blocks.size() * sizeof(CudaMeshBlock),
//                       cudaMemcpyHostToDevice, cuda_stream));

//   // Kernel call - One ThreadBlock launched per VoxelBlock
//   constexpr int kThreadsPerBlock = 8 * 32;  // Chosen at random
//   const int num_blocks = block_indices.size();
//   const float voxel_size =
//       mesh_layer->block_size() / VoxelBlock<TsdfVoxel>::kVoxelsPerSide;
//   // colorMeshBlockByClosestColorVoxel<<<num_blocks, kThreadsPerBlock, 0,
//   //                                     cuda_stream>>>(
//   //     color_block_device_ptrs,   // NOLINT
//   //     block_indices_device_ptr,  // NOLINT
//   //     mesh_layer->block_size(),  // NOLINT
//   //     voxel_size,                // NOLINT
//   //     cuda_mesh_block_device_ptrs);
//   checkCudaErrors(cudaStreamSynchronize(cuda_stream));
//   checkCudaErrors(cudaPeekAtLastError());

//   // Deallocate
//   checkCudaErrors(cudaFree(color_block_device_ptrs));
//   checkCudaErrors(cudaFree(block_indices_device_ptr));
//   checkCudaErrors(cudaFree(cuda_mesh_block_device_ptrs));
// }

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

  // NOTE(alexmillane): Generally, some of the MeshBlocks which we are
  // "coloring" will not have data in the color layer. HOWEVER, for colored
  // MeshBlocks (ie with non-empty color members), the size of the colors must
  // match vertices. Therefore we "color" all requested block_indices in two
  // parts:
  // - The first part using the color layer, and
  // - the second part a constant color.

  // Check for each index, that the MeshBlock exists, and if it does
  // allocate space for uvs
  std::vector<Index3D> block_indices;
  block_indices.reserve(requested_block_indices.size());
  std::for_each(
      requested_block_indices.begin(), requested_block_indices.end(),
      [&mesh_layer, &block_indices](const Index3D& block_idx) {
        if (mesh_layer->isBlockAllocated(block_idx)) {
          mesh_layer->getBlockAtIndex(block_idx)->expandUVsToMatchVertices();
          block_indices.push_back(block_idx);
        }
      });

  // Split block indices into two groups, one group containing indices with
  // corresponding ColorBlocks, and one without.
  std::vector<Index3D> block_indices_in_color_layer;
  std::vector<Index3D> block_indices_not_in_color_layer;
  block_indices_in_color_layer.reserve(block_indices.size());
  block_indices_not_in_color_layer.reserve(block_indices.size());
  for (const Index3D& block_idx : block_indices) {
    if (tex_layer.isBlockAllocated(block_idx)) {
      block_indices_in_color_layer.push_back(block_idx);
    } else {
      block_indices_not_in_color_layer.push_back(block_idx);
    }
  }

  // Color
  // colorMeshBlockByClosestColorVoxelGPU(
  //     tex_layer, block_indices_in_color_layer, mesh_layer, cuda_stream_);
  // colorMeshBlocksConstantGPU(block_indices_not_in_color_layer,
  //                            default_mesh_color_, mesh_layer, cuda_stream_);
}

Vector2f MeshUVIntegrator::projectToTexPatch(
    const Vector3f& vertex, const Vector3f& voxel_center,
    const float voxel_size, const TexVoxel::Dir direction) const {
  // NOTE(rasaford) since the directions encoded in TexVoxel::Dir are aligned
  // with the major coordinate axes, we do not need to do a complicated
  // projection here but can just take the respective coordinates directly
  const Vector3f texel_coords = (vertex - voxel_center) / voxel_size;
  Vector2f uv;
  switch (direction) {
    case TexVoxel::Dir::X_PLUS:
    case TexVoxel::Dir::X_MINUS:
      uv << texel_coords(1), texel_coords(2);
      break;
    case TexVoxel::Dir::Y_PLUS:
    case TexVoxel::Dir::Y_MINUS:
      uv << texel_coords(0), texel_coords(2);
      break;
    case TexVoxel::Dir::Z_PLUS:
    case TexVoxel::Dir::Z_MINUS:
      uv << texel_coords(0), texel_coords(1);
      break;
    default:
      uv << -1.0f, -1.0f;
  }
  return uv + Vector2f(0.5f, 0.5f);
}

Color MeshUVIntegrator::getDirColor(const TexVoxel::Dir dir) const {
  switch(dir) {
    case TexVoxel::Dir::X_PLUS:
    case TexVoxel::Dir::X_MINUS:
      return Color::Red();
    case TexVoxel::Dir::Y_PLUS:
    case TexVoxel::Dir::Y_MINUS:
      return Color::Green();
    case TexVoxel::Dir::Z_PLUS:
    case TexVoxel::Dir::Z_MINUS:
      return Color::Blue();
  }
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
      const TexVoxel* tex_voxel;
      // not all blocks in a layer might be allocated. So we check if a voxel
      // exists at the vertex position.
      if (getVoxelAtPosition<TexVoxel>(tex_layer, vertex, &tex_voxel)) {
        Index3D tex_block_index, tex_voxel_index;
        getBlockAndVoxelIndexFromPositionInLayer(
            tex_layer.block_size(), vertex, &tex_block_index, &tex_voxel_index);
        const Vector3f voxel_center =
            getCenterPostionFromBlockIndexAndVoxelIndex(
                tex_layer.block_size(), tex_block_index, tex_voxel_index);

        // Add the tex_voxel's colors as a patch to the MeshBlockUV.
        // NOTE(rasaford) Since getVoxelAtPosition() requires a const pointer,
        // we have to do this ugly const_cast here. In reality the
        // TexVoxel*->colors attribute is non-const anyways.
        const int patch_index = block->addPatch(
            tex_block_index, tex_voxel_index, tex_voxel->kPatchWidth,
            tex_voxel->kPatchWidth, const_cast<TexVoxel*>(tex_voxel)->colors);

        const Vector2f patch_uv = projectToTexPatch(
            vertex, voxel_center, tex_layer.voxel_size(), tex_voxel->dir);
        const Vector2f patch_uv_px = TexVoxel::kPatchWidth * patch_uv;
        Color interpolate;
        interpolation::interpolate2DLinear<Color>(
            tex_voxel->colors, patch_uv_px, TexVoxel::kPatchWidth,
            TexVoxel::kPatchWidth, &interpolate);



        // block->colors[i] = interpolate;
        block->colors[i] = getDirColor(tex_voxel->dir);
        block->uvs[i] = patch_uv;
        block->vertex_patches[i] = patch_index;

      } else {
        block->colors[i] = Color::Gray();
        block->uvs[i] = Vector2f::Zero();
      }
    }
  }
}

}  // namespace nvblox
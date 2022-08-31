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

#include "nvblox/core/common_names.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/types.h"
#include "nvblox/mesh/marching_cubes.h"
#include "nvblox/mesh/mesh_block.h"

namespace nvblox {

/**
 * @brief This class contains all functionality to construct a Mesh from the a
 * given Tsdf voxel grid. Specializations deal with adding color to the mesh.
 *
 * @tparam CudaMeshBlockType_ CudaMeshBlock type to use for transferring
 * MeshBlocks from / to the GPU
 * @tparam MeshBlockType_ MeshBlock type to use when meshing
 */
template <typename CudaMeshBlockType_, typename MeshBlockType_>
class Mesher {
 public:
  // make template parameters accessible via :: operator
  typedef CudaMeshBlockType_ CudaMeshBlockType;
  typedef MeshBlockType_ MeshBlockType;

  Mesher();
  ~Mesher();

  /// Chooses the default mesher between CPU and GPU.
  bool integrateMeshFromDistanceField(
      const TsdfLayer& distance_layer, BlockLayer<MeshBlockType>* mesh_layer,
      const DeviceType device_type = DeviceType::kGPU);

  /// Integrates only the selected blocks from the distance layer.
  bool integrateBlocksCPU(const TsdfLayer& distance_layer,
                          const std::vector<Index3D>& block_indices,
                          BlockLayer<MeshBlockType>* mesh_layer);

  bool integrateBlocksGPU(const TsdfLayer& distance_layer,
                          const std::vector<Index3D>& block_indices,
                          BlockLayer<MeshBlockType>* mesh_layer);

  // accessors for private class variables
  float min_weight() const { return min_weight_; }
  float& min_weight() { return min_weight_; }

  bool weld_vertices() const { return weld_vertices_; }
  bool& weld_vertices() { return weld_vertices_; }

 protected:
  bool isBlockMeshable(const VoxelBlock<TsdfVoxel>::ConstPtr block,
                       float cutoff) const;

  bool getTriangleCandidatesAroundVoxel(
      const VoxelBlock<TsdfVoxel>::ConstPtr block,
      const std::vector<VoxelBlock<TsdfVoxel>::ConstPtr>& neighbor_blocks,
      const Index3D& index, const Vector3f& voxel_position,
      const float voxel_size,
      marching_cubes::PerVoxelMarchingCubesResults* neighbors);

  void getTriangleCandidatesInBlock(
      const VoxelBlock<TsdfVoxel>::ConstPtr block,
      const std::vector<VoxelBlock<TsdfVoxel>::ConstPtr>& neighbor_blocks,
      const Index3D& block_index, const float block_size,
      std::vector<marching_cubes::PerVoxelMarchingCubesResults>*
          triangle_candidates);

  void getMeshableBlocksGPU(const TsdfLayer& distance_layer,
                            const std::vector<Index3D>& block_indices,
                            float cutoff_distance,
                            std::vector<Index3D>* meshable_blocks);

  void meshBlocksGPU(const TsdfLayer& distance_layer,
                     const std::vector<Index3D>& block_indices,
                     BlockLayer<MeshBlockType>* mesh_layer);

  void weldVertices(const std::vector<Index3D>& block_indices,
                    BlockLayer<MeshBlockType>* mesh_layer);

  // State.
  cudaStream_t cuda_stream_ = nullptr;

  // Offsets for cube indices.
  Eigen::Matrix<int, 3, 8> cube_index_offsets_;

  // Minimum weight to actually mesh.
  float min_weight_ = 1e-4;

  /// Whether to perform vertex welding or not. It's slow but cuts down number
  /// of vertices by 5x.
  bool weld_vertices_ = false;

  // Intermediate marching cube results.
  device_vector<marching_cubes::PerVoxelMarchingCubesResults>
      marching_cubes_results_device_;
  device_vector<int> mesh_block_sizes_device_;
  host_vector<int> mesh_block_sizes_host_;

  // These are temporary variables so we don't have to allocate every single
  // frame.
  host_vector<const VoxelBlock<TsdfVoxel>*> block_ptrs_host_;
  device_vector<const VoxelBlock<TsdfVoxel>*> block_ptrs_device_;
  host_vector<bool> meshable_host_;
  device_vector<bool> meshable_device_;
  host_vector<Vector3f> block_positions_host_;
  device_vector<Vector3f> block_positions_device_;
  host_vector<CudaMeshBlockType> mesh_blocks_host_;
  device_vector<CudaMeshBlockType> mesh_blocks_device_;

  // Caches for welding.
  device_vector<Vector3f> input_vertices_;
  device_vector<Vector3f> input_normals_;
};

/**
 * @brief MeshIntegrator for a ColorLayer, defining one color per voxel. Colors
 * are saved as vertex colors on the resulting mesh.
 *
 */
class MeshIntegrator : public Mesher<CudaMeshBlock, MeshBlock> {
 public:
  // Color mesh layer.
  // TODO(alexmillane): Currently these functions color vertices by taking the
  // CLOSEST color. Would be good to have an option at least for interpolation.
  void colorMesh(const ColorLayer& color_layer, MeshLayer* mesh_layer);
  void colorMesh(const ColorLayer& color_layer,
                 const std::vector<Index3D>& block_indices,
                 MeshLayer* mesh_layer);
  void colorMeshGPU(const ColorLayer& color_layer, MeshLayer* mesh_layer);
  void colorMeshGPU(const ColorLayer& color_layer,
                    const std::vector<Index3D>& block_indices,
                    MeshLayer* mesh_layer);
  void colorMeshCPU(const ColorLayer& color_layer, MeshLayer* mesh_layer);
  void colorMeshCPU(const ColorLayer& color_layer,
                    const std::vector<Index3D>& block_indices,
                    MeshLayer* mesh_layer);

 protected:
  // The color that the mesh takes if no coloring is available.
  const Color default_mesh_color_ = Color::Gray();
};

/**
 * @brief MeshUVIntegrator for a TexLayer. Since each voxel defines a 2D grid of
 * texture values, each vertex in the resulting mesh is assigned a uv texture
 * corrdinate in the resuling single texture.

 */
class MeshUVIntegrator : public Mesher<CudaMeshBlockUV, MeshBlockUV> {
 public:
  void textureMesh(const TexLayer& tex_layer, MeshUVLayer* mesh_layer);
  void textureMesh(const TexLayer& tex_layer,
                   const std::vector<Index3D>& block_indices,
                   MeshUVLayer* mesh_layer);
  void textureMeshGPU(const TexLayer& tex_layer, MeshUVLayer* mesh_layer);
  void textureMeshGPU(const TexLayer& tex_layer,
                      const std::vector<Index3D>& block_indices,
                      MeshUVLayer* mesh_layer);
  void textureMeshCPU(const TexLayer& tex_layer, MeshUVLayer* mesh_layer);
  void textureMeshCPU(const TexLayer& tex_layer,
                      const std::vector<Index3D>& block_indices,
                      MeshUVLayer* mesh_layer);

 protected:
  // The color that the mesh takes if no coloring is available.
  const Color default_mesh_color_ = Color::Gray();

  device_vector<Index3D> block_indices_device_;

  host_vector<const TexBlock*> tex_blocks_host_;
  device_vector<const TexBlock*> tex_blocks_device_;

  host_vector<CudaMeshBlockUV> uv_mesh_blocks_host_;
  device_vector<CudaMeshBlockUV> uv_mesh_blocks_device_;
};

}  // namespace nvblox

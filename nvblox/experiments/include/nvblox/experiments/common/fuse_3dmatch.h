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

#include <memory>
#include <string>

#include <glog/logging.h>

#include "nvblox/core/blox.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/layer_cake.h"
#include "nvblox/core/mapper.h"
#include "nvblox/core/voxels.h"
#include "nvblox/datasets/image_loader.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/projective_color_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/mesh/mesh_block.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/ray_tracing/sphere_tracer.h"

namespace nvblox {
namespace experiments {

class Fuse3DMatch {
 public:
  Fuse3DMatch() = default;
  Fuse3DMatch(const std::string& base_path,
              const std::string& timing_output_path = std::string(),
              const std::string& mesh_output_path = std::string(),
              const std::string& esdf_output_path = std::string());

  // Runs an experiment
  virtual int run();

  // Set the base path pointing to the dataset.
  virtual void setBasePath(const std::string& base_path);
  // Set the sequence number within the 3D Match dataset.
  virtual void setSequenceNumber(int sequence_num);

  // Set various settings.
  void setVoxelSize(float voxel_size);
  void setTsdfFrameSubsampling(int subsample);
  void setColorFrameSubsampling(int subsample);
  void setMeshFrameSubsampling(int subsample);
  void setEsdfFrameSubsampling(int subsample);

  // Integrate certain layers.
  virtual bool integrateFrame(const int frame_number);
  virtual bool integrateFrames();

  // Initialize the image loaders based on the current base_path and
  // sequence_num
  virtual void initializeImageLoaders(bool multithreaded = true);

  // Output a pointcloud ESDF as PLY file.
  virtual bool outputPointcloudPly() const;
  // Output a file with the mesh.
  virtual bool outputMeshPly() const;
  // Output timings to a file
  virtual bool outputTimingsToFile() const;

  // Factory: From command line args
  static Fuse3DMatch createFromCommandLineArgs(int argc, char* argv[]);

  // Get the mapper (useful for experiments where we modify mapper settings)
  RgbdMapper& mapper() { return mapper_; }

  // Dataset settings.
  std::string base_path_;
  int sequence_num_ = 1;
  int num_frames_to_integrate_ = std::numeric_limits<int>::max();
  int start_frame_ = 0;
  std::unique_ptr<datasets::ImageLoader<DepthImage>> depth_image_loader_;
  std::unique_ptr<datasets::ImageLoader<ColorImage>> color_image_loader_;

  // Params
  float voxel_size_m_ = 0.05;
  int tsdf_frame_subsampling_ = 1;
  int color_frame_subsampling_ = 1;
  int mesh_frame_subsampling_ = 1;
  int esdf_frame_subsampling_ = 1;

  // ESDF slice params
  float z_min_ = 0.5f;
  float z_max_ = 1.0f;
  float z_slice_ = 0.75f;

  // Mapper - Contains map layers and integrators
  RgbdMapper mapper_;

  // Output paths
  std::string timing_output_path_;
  std::string esdf_output_path_;
  std::string mesh_output_path_;
};

class TexFuse3DMatch : public Fuse3DMatch {
  public:
  TexFuse3DMatch() = default;
  TexFuse3DMatch(const std::string& base_path,
                 const std::string& timing_output_path = std::string(),
                 const std::string& mesh_output_path = std::string(),
                 const std::string& esdf_output_path = std::string(),
                 const std::string& texture_output_path = std::string());

  // Integrate certain layers.
  virtual bool integrateFrame(const int frame_number) override;

  void setTexFrameSubsampling(int subsample);

  // Output a file with the mesh.
  virtual bool outputMeshPly() const override;

  // Factory: From command line args
  static TexFuse3DMatch createFromCommandLineArgs(int argc, char* argv[]);

  // Get the mapper (useful for experiments where we modify mapper settings)
  TexMapper& mapper() { return mapper_; }

  int tex_frame_subsampling_ = 1;

  // Mapper - Contains map layers and integrators
  TexMapper mapper_;

  // Output paths
  std::string texture_output_path_;
};

}  //  namespace experiments
}  //  namespace nvblox

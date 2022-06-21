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
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "nvblox/experiments/common/fuse_3dmatch.h"

DECLARE_bool(alsologtostderr);

DEFINE_int32(num_frames, -1,
             "Number of frames to process. Empty means process all.");
DEFINE_int32(start_frame, 0,
             "First frame index to start integrating. Default 0.");

DEFINE_string(timing_output_path, "",
              "File in which to save the timing results.");
DEFINE_string(esdf_output_path, "",
              "File in which to save the ESDF pointcloud.");
DEFINE_string(mesh_output_path, "", "File in which to save the surface mesh.");
DEFINE_string(texture_output_path, "",
              "File in which to save the global texture.");

DEFINE_double(voxel_size, 0.0f, "Voxel resolution in meters.");
DEFINE_int32(tsdf_frame_subsampling, 0,
             "By what amount to subsample the TSDF frames. A subsample of 3 "
             "means only every 3rd frame is taken.");
DEFINE_int32(color_frame_subsampling, 0,
             "How much to subsample the color integration by.");
DEFINE_int32(mesh_frame_subsampling, 0,
             "How much to subsample the meshing by.");
DEFINE_int32(esdf_frame_subsampling, 0,
             "How much to subsample the ESDF integration by.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  // Note(rasaford) due to C==14 not supporting aggregate initialization the
  // child struct TexFuse3DMatchOptions has to initialized by hand, beware that
  // the compiler does not check that each field is assigned
  nvblox::experiments::TexFuse3DMatchOptions options;
  options.num_frames = FLAGS_num_frames;
  options.start_frame = FLAGS_start_frame;
  options.timing_output_path = FLAGS_timing_output_path;
  options.esdf_output_path = FLAGS_esdf_output_path;
  options.mesh_output_path = FLAGS_mesh_output_path;
  options.texture_output_path = FLAGS_texture_output_path;
  options.voxel_size = FLAGS_voxel_size;
  options.tsdf_frame_subsampling = FLAGS_tsdf_frame_subsampling;
  options.color_frame_subsampling = FLAGS_color_frame_subsampling;
  options.mesh_frame_subsampling = FLAGS_mesh_frame_subsampling;
  options.esdf_frame_subsampling = FLAGS_esdf_frame_subsampling;

  nvblox::experiments::TexFuse3DMatch fuser =
      nvblox::experiments::TexFuse3DMatch::createFromCommandLineArgs(argc, argv,
                                                                     options);

  return fuser.run();
}

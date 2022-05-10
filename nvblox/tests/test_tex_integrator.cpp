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
#include <gtest/gtest.h>
#include <map>
#include <opencv2/opencv.hpp>

#include "nvblox/core/accessors.h"
#include "nvblox/core/blox.h"
#include "nvblox/core/bounding_boxes.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/interpolation_3d.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/voxels.h"
#include "nvblox/integrators/projective_tex_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/primitives/primitives.h"
#include "nvblox/primitives/scene.h"

#include "nvblox/tests/gpu_image_routines.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

class TexIntegrationTest : public ::testing::Test {
 protected:
  TexIntegrationTest()
      : kSphereCenter(Vector3f(0.0f, 0.0f, 2.0f)),
        gt_layer_(voxel_size_m_, MemoryType::kUnified),
        camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {
    // Maximum distance to consider for scene generation.
    constexpr float kMaxDist = 10.0;
    constexpr float kMinWeight = 1.0;

    // Tolerance for error.
    constexpr float kDistanceErrorTolerance = truncation_distance_m_;

    // Scene is bounded to -5, -5, 0 to 5, 5, 5.
    scene.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, 0.0f),
                                          Vector3f(5.0f, 5.0f, 5.0f));
    // Create a scene with a ground plane and a sphere.
    scene.addGroundLevel(0.0f);
    scene.addCeiling(5.0f);
    scene.addPrimitive(
        std::make_unique<primitives::Sphere>(Vector3f(0.0f, 0.0f, 2.0f), 2.0f));
    // Add bounding planes at 5 meters. Basically makes it sphere in a box.
    scene.addPlaneBoundaries(-5.0f, 5.0f, -5.0f, 5.0f);

    // Get the ground truth SDF for it.
    scene.generateSdfFromScene(truncation_distance_m_, &gt_layer_);
  }

  // Scenes
  constexpr static float kSphereRadius = 2.0f;
  const Vector3f kSphereCenter;

  // Test layer
  constexpr static float voxel_size_m_ = 0.1;
  constexpr static float block_size_m_ =
      VoxelBlock<TsdfVoxel>::kVoxelsPerSide * voxel_size_m_;
  TsdfLayer gt_layer_;

  // Truncation distance
  constexpr static float truncation_distance_vox_ = 4;
  constexpr static float truncation_distance_m_ =
      truncation_distance_vox_ * voxel_size_m_;

  // Test camera
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;

  // Test Scene
  primitives::Scene scene;
};

// ProjectiveTexIntegrator child that gives the tests access to the internal
// functions.
class TestProjectiveTexIntegratorGPU : public ProjectiveTexIntegrator {
 public:
  TestProjectiveTexIntegratorGPU() : ProjectiveTexIntegrator() {}
  FRIEND_TEST(TexIntegrationTest, TruncationBandTest);
};

ColorImage generateSolidColorImage(const Color& color, const int height,
                                   const int width) {
  // Generate a random color for this scene
  ColorImage image(height, width, nvblox::MemoryType::kUnified);
  nvblox::test_utils::setImageConstantOnGpu(color, &image);
  return image;
}

std::vector<Eigen::Vector3f> getPointsOnASphere(const float radius,
                                                const Eigen::Vector3f& center,
                                                const int points_per_rad = 10) {
  std::vector<Eigen::Vector3f> sphere_points;
  for (int azimuth_idx = 0; azimuth_idx < 2 * points_per_rad; azimuth_idx++) {
    for (int elevation_idx = 0; elevation_idx < points_per_rad;
         elevation_idx++) {
      const float azimuth = azimuth_idx * M_PI / points_per_rad - M_PI;
      const float elevation =
          elevation_idx * M_PI / points_per_rad - M_PI / 2.0f;
      Eigen::Vector3f p =
          radius * Eigen::Vector3f(cos(azimuth) * sin(elevation),
                                   sin(azimuth) * sin(elevation),
                                   cos(elevation));
      p += center;
      sphere_points.push_back(p);
    }
  }
  return sphere_points;
}

float checkSphereColor(const TexLayer& tex_layer, const Vector3f& center,
                       const float radius, const Color& color) {
  // Check that each sphere is colored appropriately (if observed)
  int num_observed = 0;
  int num_tested = 0;
  auto check_color = [&num_tested, &num_observed](
                         const TexVoxel& voxel, const Color& color_2) -> void {
    ++num_tested;
    if (voxel.weight >= 1.0f) {
      for (int i = 0; i < TexVoxel::kPatchWidth; ++i) {
        for (int j = 0; j < TexVoxel::kPatchWidth; ++j) {
          EXPECT_EQ(voxel(i, j), color_2);
        }
      }
      ++num_observed;
    }
  };

  const std::vector<Eigen::Vector3f> sphere_points =
      getPointsOnASphere(radius, center);
  for (const Vector3f p : sphere_points) {
    const TexVoxel* color_voxel;
    EXPECT_TRUE(getVoxelAtPosition<TexVoxel>(tex_layer, p, &color_voxel));
    check_color(*color_voxel, color);
  }

  const float ratio_observed_points =
      static_cast<float>(num_observed) / static_cast<float>(num_tested);
  return ratio_observed_points;
}

// TEST_F(TexIntegrationTest, TruncationBandTest) {
//   // Check the GPU version against a hand-rolled CPU implementation.
//   TestProjectiveTexIntegratorGPU integrator;

//   // The distance from the surface that we "pass" blocks within.
//   constexpr float kTestDistance = voxel_size_m_;

//   std::vector<Index3D> all_indices = gt_layer_.getAllBlockIndices();
//   std::vector<Index3D> valid_indices =
//       integrator.reduceBlocksToThoseInTruncationBand(all_indices, gt_layer_,
//                                                      kTestDistance);
//   integrator.finish();

//   // Horrible N^2 complexity set_difference implementation. But easy to write
//   std::vector<Index3D> not_valid_indices;
//   for (const Index3D& idx : all_indices) {
//     if (std::find(valid_indices.begin(), valid_indices.end(), idx) ==
//         valid_indices.end()) {
//       not_valid_indices.push_back(idx);
//     }
//   }

//   // Check indices touching band
//   for (const Index3D& idx : valid_indices) {
//     const auto block_ptr = gt_layer_.getBlockAtIndex(idx);
//     bool touches_band = false;
//     auto touches_band_lambda = [&touches_band, kTestDistance](
//                                    const Index3D& voxel_index,
//                                    const TsdfVoxel* voxel) -> void {
//       if (std::abs(voxel->distance) <= kTestDistance) {
//         touches_band = true;
//       }
//     };
//     callFunctionOnAllVoxels<TsdfVoxel>(*block_ptr, touches_band_lambda);
//     EXPECT_TRUE(touches_band);
//   }

//   // Check indices NOT touching band
//   for (const Index3D& idx : not_valid_indices) {
//     const auto block_ptr = gt_layer_.getBlockAtIndex(idx);
//     bool touches_band = false;
//     auto touches_band_lambda = [&touches_band, kTestDistance](
//                                    const Index3D& voxel_index,
//                                    const TsdfVoxel* voxel) -> void {
//       if (std::abs(voxel->distance) <= kTestDistance) {
//         touches_band = true;
//       }
//     };
//     callFunctionOnAllVoxels<TsdfVoxel>(*block_ptr, touches_band_lambda);
//     EXPECT_FALSE(touches_band);
//   }
// }

TEST_F(TexIntegrationTest, IntegrateTexToGroundTruthDistanceField) {
  // Create an integrator.
  ProjectiveTexIntegrator tex_integrator;

  // Simulate a trajectory of the requisite amount of points, on the circle
  // around the sphere.
  constexpr float kTrajectoryRadius = 4.0f;
  constexpr float kTrajectoryHeight = 2.0f;
  constexpr int kNumTrajectoryPoints = 80;
  constexpr float radians_increment = 2 * M_PI / kNumTrajectoryPoints;

  // Allocated layer in unified memory
  TexLayer tex_layer(voxel_size_m_, MemoryType::kUnified);

  // Generate a random color for this scene
  const Color color = Color::Red();
  const ColorImage image = generateSolidColorImage(color, height_, width_);

  // Set keeping track of which blocks were touched during the test
  Index3DSet touched_blocks;

  for (size_t i = 0; i < kNumTrajectoryPoints; i++) {
    const float theta = radians_increment * i;
    // Convert polar to cartesian coordinates.
    Vector3f cartesian_coordinates(kTrajectoryRadius * std::cos(theta),
                                   kTrajectoryRadius * std::sin(theta),
                                   kTrajectoryHeight);
    // The camera has its z axis pointing towards the origin.
    Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
    Eigen::Quaternionf rotation_theta(
        Eigen::AngleAxisf(M_PI + theta, Vector3f::UnitZ()));

    // Construct a transform from camera to scene with this.
    Transform T_S_C = Transform::Identity();
    T_S_C.prerotate(rotation_theta * rotation_base);
    T_S_C.pretranslate(cartesian_coordinates);

    // Generate an image with a single color
    std::vector<Index3D> updated_blocks;
    tex_integrator.integrateFrame(image, T_S_C, camera_, gt_layer_, &tex_layer,
                                  &updated_blocks);
    // Accumulate touched block indices
    std::copy(updated_blocks.begin(), updated_blocks.end(),
              std::inserter(touched_blocks, touched_blocks.end()));
  }

  // Lambda that checks if voxels have the passed color (if they have weight >
  // 0)
  auto color_check_lambda = [&color](const Index3D& voxel_idx,
                                     const TexVoxel* voxel) -> void {
    if (voxel->weight > 0.0f) {
      for (size_t col = 0; col < voxel->kPatchWidth; ++col) {
        for (size_t row = 0; row < voxel->kPatchWidth; ++row) {
          EXPECT_EQ((*voxel)(row, col), color);
        }
      }
    }
  };

  // Check that all touched blocks are the color we chose
  for (const Index3D& block_idx : touched_blocks) {
    callFunctionOnAllVoxels<TexVoxel>(*tex_layer.getBlockAtIndex(block_idx),
                                      color_check_lambda);
  }

  // check that all TexVoxel directions are equally frequent in the textured
  // sphere. Tolerance defines the minimal ratio of the least frequent direction
  // to the most frequent one
  const float dir_tolerance = 0.5;
  auto dir_frequencies = std::unordered_map<TexVoxel::Dir, int>();
  auto populate_dir_frequencies = [&dir_frequencies](
                                      const Index3D& voxel_idx,
                                      const TexVoxel* voxel) -> void {
    if (dir_frequencies.find(voxel->dir) != dir_frequencies.end()) {
      dir_frequencies[voxel->dir]++;
    } else {
      dir_frequencies[voxel->dir] = 1;
    }
  };
  for (const Index3D& block_idx : touched_blocks) {
    callFunctionOnAllVoxels<TexVoxel>(*tex_layer.getBlockAtIndex(block_idx),
                                      populate_dir_frequencies);
  }
  int min_freq = 1000000, max_freq = 0;
  for (const auto& item : dir_frequencies) {
    if (item.first == TexVoxel::Dir::NONE) continue;
    min_freq = MIN(min_freq, item.second);
    max_freq = MAX(max_freq, item.second);
  }
  EXPECT_GT(min_freq, 0);
  EXPECT_GT(static_cast<float>(min_freq) / max_freq, dir_tolerance);

  // Check that most points on the surface of the sphere have been observed
  int num_points_on_sphere_surface_observed = 0;
  const std::vector<Eigen::Vector3f> sphere_points =
      getPointsOnASphere(kSphereRadius, kSphereCenter);
  const int num_surface_points_tested = sphere_points.size();
  for (const Vector3f p : sphere_points) {
    const TexVoxel* tex_voxel;
    EXPECT_TRUE(getVoxelAtPosition<TexVoxel>(tex_layer, p, &tex_voxel));
    if (tex_voxel->weight > 0.0f) {
      ++num_points_on_sphere_surface_observed;
    }
  }
  const float ratio_observed_surface_points =
      static_cast<float>(num_points_on_sphere_surface_observed) /
      static_cast<float>(num_surface_points_tested);
  std::cout << "num_points_on_sphere_surface_observed: "
            << num_points_on_sphere_surface_observed << std::endl;
  std::cout << "num_surface_points_tested: " << num_surface_points_tested
            << std::endl;
  std::cout << "ratio_observed_surface_points: "
            << ratio_observed_surface_points << std::endl;
  EXPECT_GT(ratio_observed_surface_points, 0.5);

  // Check that all color blocks have a corresponding block in the tsdf layer
  for (const Index3D& block_idx : tex_layer.getAllBlockIndices()) {
    EXPECT_NE(gt_layer_.getBlockAtIndex(block_idx), nullptr);
  }

  // TODO (rasaford) Meshing currently does not work with TexVoxels yet.
  // Implement at least some approximation of full uv unwrapping ASAP
  // Generate a mesh from the "reconstruction"
  MeshUVIntegrator mesh_integrator;
  MeshUVLayer mesh_layer(block_size_m_, MemoryType::kUnified);
  EXPECT_TRUE(
      mesh_integrator.integrateMeshFromDistanceField(gt_layer_, &mesh_layer));
  mesh_integrator.textureMeshCPU(tex_layer, &mesh_layer);

  // Write to file
  auto textured_mesh = io::packTextures(mesh_layer);
  io::outputMeshToPly(textured_mesh->mesh, "tex_sphere_mesh.ply");
  cv::imwrite("tex_sphere_mesh.png", textured_mesh->texture);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

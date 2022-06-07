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

#include "nvblox/core/indexing.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/types.h"
#include "nvblox/core/voxels.h"

#include "nvblox/tex/surface_estimation.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include "nvblox/tests/gpu_indexing.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;
using namespace nvblox::tex;

constexpr float kFloatEpsilon = 1e-4;

class TexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::srand(0);
    block_size_ = VoxelBlock<bool>::kVoxelsPerSide * voxel_size_;
  }

  float block_size_;
  float voxel_size_ = 0.05;
};

TEST_F(TexTest, BijectiveNeighborIndexing) {
  std::vector<int> indices;

  for (int i = 0; i < neighbor::kCubeNeighbors; i++) {
    Index3D offset = neighbor::blockOffsetFromNeighborIndex(i);
    int linear_idx = neighbor::neighborBlockIndexFromOffset(offset);

    // check that the offset is in the range [-1, 1] for each coordinate
    // clang-format off
    EXPECT_LE(-1, offset[0]); EXPECT_LE(offset[0], 1);
    EXPECT_LE(-1, offset[1]); EXPECT_LE(offset[1], 1);
    EXPECT_LE(-1, offset[2]); EXPECT_LE(offset[2], 1);
    // clang-format on
    // check that the concatenation of both indexing operations is the identity
    EXPECT_EQ(i, linear_idx);
    indices.push_back(linear_idx);
  }

  // remove duplicate elements from the indices vector
  std::sort(indices.begin(), indices.end());
  std::unique(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  // check that we cover all linear indices, i.e. the mapping is bijective
  EXPECT_EQ(indices.size(), neighbor::kCubeNeighbors);
}

// TEST_F(TexTest, GetVoxelIndex) {
//   ASSERT_NEAR(voxel_size_, 0.05, kFloatEpsilon);
//   ASSERT_NEAR(block_size_, 0.4, kFloatEpsilon);

//   Vector3f point(0, 0, 0);
//   Index3D voxel_index = getVoxelIndexFromPositionInLayer(block_size_, point);
//   EXPECT_TRUE(voxel_index == Index3D(0, 0, 0));

//   point = Vector3f(block_size_, block_size_, block_size_);
//   voxel_index = getVoxelIndexFromPositionInLayer(block_size_, point);
//   EXPECT_TRUE(voxel_index == Index3D(0, 0, 0));

//   point = Vector3f(2.0f, 1.0f, 3.0f);
//   voxel_index = getVoxelIndexFromPositionInLayer(block_size_, point);
//   EXPECT_TRUE(voxel_index == Index3D(0, 4, 4)) << voxel_index;

//   point = Vector3f(-2.0f, -1.0f, -3.0f);
//   voxel_index = getVoxelIndexFromPositionInLayer(block_size_, point);
//   EXPECT_TRUE(voxel_index == Index3D(0, 4, 4)) << voxel_index;
// }

// TEST_F(TexTest, CenterIndexing) {
//   AlignedVector<Vector3f> test_points;
//   test_points.push_back({0, 0, 0});
//   test_points.push_back({0.05, 0.05, 0.05});
//   test_points.push_back({0.025, 0.025, 0.025});
//   test_points.push_back({-1, -3, 2});

//   for (const Vector3f& point : test_points) {
//     Index3D voxel_index = getVoxelIndexFromPositionInLayer(block_size_,
//     point); Index3D block_index =
//     getBlockIndexFromPositionInLayer(block_size_, point);

//     Vector3f reconstructed_point = getPositionFromBlockIndexAndVoxelIndex(
//         block_size_, block_index, voxel_index);
//     Vector3f reconstructed_center_point =
//         getCenterPostionFromBlockIndexAndVoxelIndex(block_size_, block_index,
//                                                     voxel_index);

//     // Check the difference between the corner and the center.
//     Vector3f center_difference =
//         reconstructed_center_point - reconstructed_point;

//     // Needs to be within voxel size.
//     EXPECT_LT((reconstructed_point - point).cwiseAbs().maxCoeff(),
//     voxel_size_); EXPECT_LT((reconstructed_center_point -
//     point).cwiseAbs().maxCoeff(),
//               voxel_size_);
//     EXPECT_TRUE(center_difference.isApprox(
//         Vector3f(voxel_size_ / 2.0f, voxel_size_ / 2.0f, voxel_size_ / 2.0f),
//         kFloatEpsilon));
//   }
// }

// TEST_F(TexTest, getBlockAndVoxelIndexFromPositionInLayer) {
//   constexpr int kNumTests = 1000;
//   for (int i = 0; i < kNumTests; i++) {
//     // Random block and voxel indices
//     constexpr int kRandomBlockIndexRange = 1000;
//     const Index3D block_index = test_utils::getRandomIndex3dInRange(
//         -kRandomBlockIndexRange, kRandomBlockIndexRange);
//     const Index3D voxel_index = test_utils::getRandomIndex3dInRange(
//         0, VoxelBlock<bool>::kVoxelsPerSide - 1);

//     // 3D point from indices, including sub-voxel randomness
//     constexpr float voxel_size = 0.1;
//     constexpr float block_size = VoxelBlock<bool>::kVoxelsPerSide *
//     voxel_size; const Vector3f delta =
//         test_utils::getRandomVector3fInRange(0.0f, voxel_size);
//     const Vector3f p_L = (block_index.cast<float>() * block_size) +
//                          (voxel_index.cast<float>() * voxel_size) + delta;

//     // Point back to voxel indices
//     Index3D block_index_check;
//     Index3D voxel_index_check;
//     getBlockAndVoxelIndexFromPositionInLayer(
//         block_size, p_L, &block_index_check, &voxel_index_check);

//     // Check we get out what we put in
//     EXPECT_EQ(block_index_check.x(), block_index.x());
//     EXPECT_EQ(block_index_check.y(), block_index.y());
//     EXPECT_EQ(block_index_check.z(), block_index.z());
//   }
// }

// TEST_F(TexTest, getBlockAndVoxelIndexFromPositionInLayerRoundingErrors) {
//   constexpr int kNumTests = 1e7;
//   std::vector<Vector3f> positions;
//   positions.reserve(kNumTests);
//   for (int i = 0; i < kNumTests; i++) {
//     positions.push_back(Vector3f::Random());
//   }

//   std::vector<Index3D> block_indices;
//   std::vector<Index3D> voxel_indices;
//   constexpr float kBlockSize = 0.1;
//   test_utils::getBlockAndVoxelIndexFromPositionInLayerOnGPU(
//       kBlockSize, positions, &block_indices, &voxel_indices);

//   for (int i = 0; i < kNumTests; i++) {
//     EXPECT_TRUE((voxel_indices[i].array() >= 0).all());
//     EXPECT_TRUE(
//         (voxel_indices[i].array() < VoxelBlock<bool>::kVoxelsPerSide).all());
//   }
// }

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

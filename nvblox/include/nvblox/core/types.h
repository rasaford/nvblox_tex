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

#include <iostream>

#include <cuda_runtime.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace nvblox {

// Whether the storage or processing is happening on CPU, GPU, or any future
// amazing hardware- accelerated platform.
enum class DeviceType { kCPU, kGPU };

// How GPU data is stored, either in Device-only, unified (both) memory or in an
// ObjectPool. CAUTION: When unsing kPool a unified_ptr of this type will
// degenerate into a weak_ptr (observing-only). Since such a pointer does not
// manage it's own memory but is only given a small section of a larger block.
// TODO(rasaford): fix the above behavior to keep all pointer types consistent.
// NOTE(alexmillane): tag: c++17, switch to constexpr when we move to c++17.
// clang-format off
enum class MemoryType { kDevice, kUnified, kHost, kPool };
inline std::string toString(MemoryType memory_type) {
  switch (memory_type) {
    case MemoryType::kDevice: return "kDevice"; break;
    case MemoryType::kUnified: return "kUnified"; break;
    case MemoryType::kHost: return "kHost"; break;
    case MemoryType::kPool: return "kPool"; break;
    // unreachable code:
    default: return "undefined_ptr"; break;
  }
}
// clang-format on

// define eigen array indices to be 32 bit for better performance on cuda.
// For more details see: https://eigen.tuxfamily.org/dox/TopicCUDA.html
// #define EIGEN_DEFAULT_DENSE_INDEX_TYPE int;
typedef Eigen::Vector3i Index3D;
typedef Eigen::Vector2i Index2D;
typedef Eigen::Matrix<int, 6, 1> Index6D;

typedef Eigen::Vector3f Vector3f;
typedef Eigen::Vector2f Vector2f;

typedef Eigen::AlignedBox3f AxisAlignedBoundingBox;

typedef Eigen::Isometry3f Transform;

typedef unsigned char uchar;

// Aligned Eigen containers
template <typename Type>
using AlignedVector = std::vector<Type, Eigen::aligned_allocator<Type>>;

enum class InterpolationType { kNearestNeighbor, kLinear, kMax, kMin };

}  // namespace nvblox

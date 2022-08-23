// This allocator is based on
// https://www.codeproject.com/Articles/746630/O-Object-Pool-in-Cplusplus
#pragma once

#include <memory.h>
#include <vector>
#include "nvblox/core/blox.h"
#include "nvblox/core/hash.h"
#include "nvblox/core/traits.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"

namespace nvblox {
namespace alloc {

template <typename T>
class HostAllocator {
 public:
  static inline T* allocate(size_t size, int device) {
    T* ptr;
    // NOTE(rasaford): Allocate host memory and map it to the GPU, such that it
    // is able to access it as well (Zero-Copy). This is slower than direcly
    // allcating the memory on the GPU since we are limited by PCIe speeds.
    checkCudaErrors(cudaMallocHost(&ptr, size));
    // checkCudaErrors(
    //     cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device));
    // checkCudaErrors(cudaMemAdvise(ptr, size,
    // cudaMemAdviseSetPreferredLocation,
    //                               cudaCpuDeviceId));
    return ptr;
  }
  static inline void deallocate(T* ptr) {
    checkCudaErrors(
        cudaFreeHost(const_cast<void*>(reinterpret_cast<void const*>(ptr))));
  }
};

template <typename T>
class DeviceAllocator {
 public:
  static inline T* allocate(size_t size, int device) {
    T* ptr;
    // NOTE(rasaford): Allocate the memory on the GPU directly. Setting the
    // preferred location, instructs the driver to keep this memory on the GPU
    // if possible. If not, it can however still be swapped out to Host memory.
    checkCudaErrors(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
    checkCudaErrors(
        cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device));
    return ptr;
  }
  static inline void deallocate(T* ptr) { checkCudaErrors(cudaFree(ptr)); }
};

// Memory allocator for Device and host memory
// TODO: this pool will never free up memory, unless it is destroyed
template <typename BlockType, class Allocator>
class ObjectPool {
 public:
  typedef typename Index3DHashMapType<BlockType*>::type BlockHash;

  ObjectPool(int device, cudaStream_t stream, int initial_capacity = 1 << 9,
             size_t max_block_length = 1 << 12)
      : first_deleted_(nullptr),
        count_in_block_(0),
        block_capacity_(initial_capacity),
        first_block_(initial_capacity, device),
        max_block_length_(max_block_length),
        device(device),
        stream(stream) {
    if (max_block_length < 1) {
      throw std::invalid_argument("max_block_length must be at last 1");
    }

    block_memory_ = first_block_.memory;
    last_block_ = &first_block_;
  };

  ~ObjectPool() {
    MemoryBlock* block = first_block_.next_block;
    while (block != nullptr) {
      MemoryBlock* next = block->next_block;
      delete block;
      block = next;
    }
  }

  BlockType* alloc(Index3D index) {
    BlockType* address;
    BlockType* result;

    if (first_deleted_ != nullptr) {
      address = first_deleted_;
      first_deleted_ = *((BlockType**)first_deleted_);
      // TODO: only do new when we are on the host
      result = new (address) BlockType();
    } else {
      if (count_in_block_ >= block_capacity_) {
        allocate_block();
      }
      address = block_memory_;
      address += count_in_block_;
      count_in_block_++;
      // TODO: only do new when we are on the host
      result = new (address) BlockType();
    }

    block_hash_.emplace(index, result);
    return result;
  }

  void dealloc(Index3D index, BlockType* ptr) {
    ptr->~BlockType();
    block_hash_.erase(index);
    *((BlockType**)ptr) = first_deleted_;
    first_deleted_ = ptr;
  }

  BlockType* isAllocated(Index3D index) {
    auto it = block_hash_.find(index);
    return it == block_hash_.end() ? nullptr : it->second;
  }

 private:
  struct MemoryBlock {
    BlockType* memory;
    size_t capacity;
    int device;
    MemoryBlock* next_block;

    MemoryBlock(size_t capacity, int device)
        : capacity(capacity), device(device), next_block(nullptr) {
      if (capacity < 1) {
        throw std::invalid_argument("capacity has to be greater than 1");
      }
      memory = Allocator::allocate(sizeof(BlockType) * capacity, device);
      if (memory == nullptr) {
        throw std::bad_alloc();
      }
    }

    ~MemoryBlock() { Allocator::deallocate(memory); }
  };

 protected:
  void allocate_block() {
    size_t size = count_in_block_;
    if (size >= max_block_length_) {
      size = max_block_length_;
    } else {
      size *= 2;

      if (size < count_in_block_) {
        std::overflow_error("allocation is too big");
      }
      if (size >= max_block_length_) {
        size = max_block_length_;
      }
    }

    std::cout << "allocate new block " << size << std::endl;
    // create a new memory block and update the internal structure
    MemoryBlock* new_block = new MemoryBlock(size, device);
    last_block_->next_block = new_block;
    last_block_ = new_block;
    block_memory_ = new_block->memory;
    count_in_block_ = 0;
    block_capacity_ = size;
  }

  // allocator management
  BlockType* block_memory_;
  BlockType* first_deleted_;
  size_t count_in_block_;
  size_t block_capacity_;
  MemoryBlock first_block_;
  MemoryBlock* last_block_;
  size_t max_block_length_;

  BlockHash block_hash_;

  cudaStream_t stream;
  int device;
};

template <typename BlockType>
class Allocator {
 public:
  typedef ObjectPool<BlockType, HostAllocator<BlockType>> HostPool;
  typedef ObjectPool<BlockType, DeviceAllocator<BlockType>> DevicePool;

  Allocator() {
    cudaGetDevice(&device);
    cudaStreamCreate(&stream);
    host_pool = std::make_unique<HostPool>(device, stream);
    device_pool = std::make_unique<DevicePool>(device, stream);
  };

  ~Allocator() { cudaStreamDestroy(stream); }

  BlockType* toDevice(const Index3D& index) {
    // find in host pool
    BlockType* host_block = host_pool->isAllocated(index);
    BlockType* device_block = device_pool->alloc(index);

    if (host_block != nullptr) {
      cudaMemcpy(device_block, host_block, sizeof(BlockType),
                 cudaMemcpyDefault);
      host_pool->dealloc(index, host_block);
    }
    return device_block;
  }

  BlockType* toHost(const Index3D& index) {
    BlockType* device_block = device_pool->isAllocated(index);
    BlockType* host_block = host_pool->alloc(index);

    if (device_block != nullptr) {
      cudaMemcpy(host_block, device_block, sizeof(BlockType),
                 cudaMemcpyDefault);
      device_pool->dealloc(index, device_block);
    }
    return host_block;
  }

 protected:
  std::unique_ptr<HostPool> host_pool;
  std::unique_ptr<DevicePool> device_pool;

  cudaStream_t stream;
  int device;
};

}  // namespace alloc
}  // namespace nvblox
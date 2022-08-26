// This allocator is based on
// https://www.codeproject.com/Articles/746630/O-Object-Pool-in-Cplusplus
#pragma once

#include <memory.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include "nvblox/core/blox.h"
#include "nvblox/core/hash.h"
#include "nvblox/core/traits.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/utils/timing.h"

namespace nvblox {
namespace alloc {

template <typename T>
class HostAllocator {
 public:
  static inline T* allocate(const size_t size, const int device,
                            const cudaStream_t& stream) {
    T* ptr;
    // NOTE(rasaford): Allocate Host memory by allocating Unified Memory and
    // setting the preferred location to the CPU. For more details see:
    // https://developer.nvidia.com/blog/improving-gpu-memory-oversubscription-performance/
    checkCudaErrors(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
    checkCudaErrors(cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, stream));
    // checkCudaErrors(cudaMemsetAsync(ptr, 0, size, stream));
    checkCudaErrors(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation,
                                  cudaCpuDeviceId));
    // NOTE(rasaford): Create a mapping for the memory on the device, such that
    // it can access it directly
    checkCudaErrors(
        cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device));
    return ptr;
  }
  static inline void deallocate(T* ptr) {
    // Strip any possible const from the given type
    checkCudaErrors(cudaFree(ptr));
  }
};

template <typename T>
class DeviceAllocator {
 public:
  static inline T* allocate(const size_t size, const int device,
                            const cudaStream_t& stream) {
    T* ptr;
    // NOTE(rasaford): Allocate the memory on the GPU directly. Setting the
    // preferred location and prefetching initially, instructs the driver to
    // keep this memory on the GPU if possible. If not, it can however still be
    // swapped out to Host memory.
    checkCudaErrors(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
    checkCudaErrors(cudaMemPrefetchAsync(ptr, size, device, stream));
    // checkCudaErrors(cudaMemsetAsync(ptr, 0, size, stream));
    checkCudaErrors(
        cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device));
    return ptr;
  }
  static inline void deallocate(T* ptr) {
    // Strip any possible const from the given type
    checkCudaErrors(cudaFree(ptr));
  }
};

struct FreeBlock {
  void* next_free = nullptr;
  void* base_block = nullptr;

  inline void reset() {
    next_free = nullptr;
    base_block = nullptr;
  }
};

// Memory allocator for Device and host memory
// TODO: this pool will never free up memory, unless it is destroyed
template <typename BlockType, class Allocator>
class ObjectPool {
 public:
  ObjectPool(int device, cudaStream_t stream, int initial_capacity = 1 << 9,
             size_t max_block_length = 1 << 12)
      : first_deleted_(nullptr),
        first_block_(initial_capacity, device, stream),
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

  BlockType* alloc() {
    BlockType* address;
    MemoryBlock* base_block = last_block_;

    if (first_deleted_ != nullptr) {
      base_block = reinterpret_cast<MemoryBlock*>(first_deleted_->base_block);
      FreeBlock* free_ptrs_block =
          reinterpret_cast<FreeBlock*>(base_block->free_ptrs);
      FreeBlock* next_free_block =
          reinterpret_cast<FreeBlock*>(first_deleted_->next_free);

      size_t block_idx = first_deleted_ - free_ptrs_block;
      if (block_idx >= base_block->capacity) {
        throw std::runtime_error("Block idx must be in MemoryBlock* capacity");
      }

      address = base_block->memory + block_idx;
      FreeBlock* tmp = first_deleted_;
      first_deleted_ = next_free_block;
      tmp->reset();
    } else {
      if (base_block->count >= base_block->capacity) {
        base_block = allocate_block();
      }
      address = block_memory_ + base_block->count;
      base_block->count++;
    }

    if (address < base_block->memory ||
        address >= base_block->memory + base_block->capacity) {
      throw std::runtime_error("invalid allocation ptr out of range");
    }

    // store the allocated ptr for referencing in dealloc
    allocated_[address] = base_block;
    return address;
  }

  void dealloc(BlockType* pointer) {
    // NOTE(rasaford): This block might be on the GPU, so we don't call the
    // destructor on it, but just free the block.  ptr->~BlockType();
    const auto it = allocated_.find(pointer);
    // NOTE(rasaford): Check that this BlockType* has been allocated. Throw to
    // prevent double free.
    if (it == allocated_.end()) {
      throw std::runtime_error("ptr has not been allocated");
    }

    // update the free block of *ptr to reflect the deallocated state
    MemoryBlock* base_block = it->second;
    if (base_block->memory > pointer ||
        pointer > base_block->memory + base_block->capacity) {
      throw std::runtime_error(
          "dealloc: BlockType* is not in the base block memory range");
    }

    const size_t block_idx = pointer - base_block->memory;
    FreeBlock* new_free_block = &base_block->free_ptrs[block_idx];

    if (new_free_block > base_block->free_ptrs + base_block->capacity) {
      throw std::runtime_error(
          "dealloc: FreeBlock* is not in block memory range");
    }

    new_free_block->next_free = reinterpret_cast<void*>(first_deleted_);
    new_free_block->base_block = reinterpret_cast<void*>(base_block);
    base_block->count--;
    first_deleted_ = new_free_block;

    allocated_.erase(pointer);
  }

  bool isAllocated(BlockType* pointer) const {
    return allocated_.find(pointer) != allocated_.end();
  }

 private:
  struct MemoryBlock {
    // Memory managed by this Block
    BlockType* memory;
    // Array of FreeBlock for each BlockType objects that could be allocated
    FreeBlock* free_ptrs;

    int count;
    size_t capacity;

    MemoryBlock* next_block;

    MemoryBlock(size_t capacity, int device, const cudaStream_t& stream)
        : count(0), capacity(capacity), next_block(nullptr) {
      if (capacity < 1) {
        throw std::invalid_argument("capacity has to be greater than 1");
      }
      memory =
          Allocator::allocate(sizeof(BlockType) * capacity, device, stream);
      if (memory == nullptr) {
        throw std::bad_alloc();
      }
      free_ptrs = new FreeBlock[capacity];
    }

    ~MemoryBlock() {
      Allocator::deallocate(memory);
      delete[] free_ptrs;
    }
  };

 protected:
  MemoryBlock* allocate_block() {
    size_t size = last_block_->count;
    if (size < max_block_length_) {
      size *= 2;
      if (size < last_block_->count) {
        std::overflow_error(
            "allocation is too big, resulting in MemoryBlock::size overflow");
      }
    }
    size = std::min(size, max_block_length_);

    // create a new memory block and update the internal management structure
    MemoryBlock* new_block = new MemoryBlock(size, device, stream);
    last_block_->next_block = new_block;
    last_block_ = new_block;
    block_memory_ = new_block->memory;

    return new_block;
  }

  // allocator management
  MemoryBlock first_block_;
  MemoryBlock* last_block_;
  BlockType* block_memory_;
  FreeBlock* first_deleted_;
  size_t max_block_length_;
  // allocated index keeping track of live allocations
  std::unordered_map<BlockType*, MemoryBlock*> allocated_;
  // CUDA parameters
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

  typename BlockType::Ptr toDevice(typename BlockType::Ptr host_block) {
    timing::Timer to_device_timer("prefetch/prefetch/to_device");
    BlockType* device_ptr = device_pool->alloc();
    BlockType* host_ptr = host_block.get();

    if (host_ptr != nullptr && host_pool->isAllocated(host_ptr)) {
      timing::Timer to_device_dealloc("prefetch/prefetch/dealloc");
      cudaMemcpyAsync(device_ptr, host_ptr, sizeof(BlockType),
                      cudaMemcpyDefault, stream);
      host_pool->dealloc(host_ptr);
      to_device_dealloc.Stop();
    }
    to_device_timer.Stop();
    return unified_ptr<BlockType>(device_ptr, MemoryType::kPool);
  }

  typename BlockType::Ptr toHost(typename BlockType::Ptr device_block) {
    timing::Timer to_host_timer("prefetch/evict/to_host");
    BlockType* host_ptr = host_pool->alloc();
    BlockType* device_ptr = device_block.get();

    if (device_ptr != nullptr && device_pool->isAllocated(device_ptr)) {
      timing::Timer to_host_memcpy("prefetch/evict/memcpy");
      cudaMemcpyAsync(host_ptr, device_ptr, sizeof(BlockType),
                      cudaMemcpyDefault, stream);
      to_host_memcpy.Stop();
      timing::Timer to_host_dealloc("prefetch/evict/dealloc");
      device_pool->dealloc(device_ptr);
      to_host_dealloc.Stop();
    }
    to_host_timer.Stop();
    return unified_ptr<BlockType>(host_ptr, MemoryType::kPool);
  }

 protected:
  std::unique_ptr<HostPool> host_pool;
  std::unique_ptr<DevicePool> device_pool;

  cudaStream_t stream;
  int device;
};

}  // namespace alloc
}  // namespace nvblox
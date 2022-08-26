// This allocator is based on
// https://www.codeproject.com/Articles/746630/O-Object-Pool-in-Cplusplus
#pragma once

#include <memory.h>
#include <unordered_map>
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
  static inline T* allocate(const size_t size, const int device,
                            const cudaStream_t& stream) {
    T* ptr;
    // NOTE(rasaford): Allocate Host memory by allocating Unified Memory and
    // setting the preferred location to the CPU. For more details see:
    // https://developer.nvidia.com/blog/improving-gpu-memory-oversubscription-performance/
    checkCudaErrors(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
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
    checkCudaErrors(
        cudaFree(const_cast<void*>(reinterpret_cast<void const*>(ptr))));
  }
};

template <typename T>
class DeviceAllocator {
 public:
  static inline T* allocate(const size_t size, const int device,
                            const cudaStream_t& stream) {
    T* ptr;
    // NOTE(rasaford): Allocate the memory on the GPU directly. Setting the
    // preferred location and prefetching initially, instructs the driver to keep
    // this memory on the GPU if possible. If not, it can however still be
    // swapped out to Host memory.
    checkCudaErrors(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
    checkCudaErrors(cudaMemPrefetchAsync(ptr, size, device, stream));
    checkCudaErrors(
        cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device));
    return ptr;
  }
  static inline void deallocate(T* ptr) {
    // Strip any possible const from the given type
    checkCudaErrors(
        cudaFree(const_cast<void*>(reinterpret_cast<void const*>(ptr))));
  }
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

  BlockType* alloc(Index3D index) {
    BlockType* address;
    MemoryBlock* base_block = last_block_;

    if (first_deleted_ != nullptr) {
      base_block = reinterpret_cast<MemoryBlock*>(first_deleted_->base_block);
      size_t block_idx =
          first_deleted_ - reinterpret_cast<FreeBlock*>(base_block->free_ptrs);
      if (block_idx >= base_block->capacity) {
        throw std::runtime_error("Block idx must be in MemoryBlock* capacity");
      }

      address = base_block->memory + block_idx;
      FreeBlock* tmp = first_deleted_;
      first_deleted_ = reinterpret_cast<FreeBlock*>(first_deleted_->next_free);
      tmp->reset();
    } else {
      if (count_in_block_ >= block_capacity_) {
        allocate_block();
        base_block = last_block_;
      }
      if (count_in_block_ > base_block->capacity) {
        throw std::runtime_error(
            "Block address must be in MemoryBlock* capacity");
      }
      address = block_memory_ + count_in_block_;
      count_in_block_++;
    }

    // The allocated memory is not initialized. I.e. there could be another
    // valid block there, if we did not write anything BlockType* result = new
    // new (address) BlockType();

    if (address < base_block->memory ||
        address >= base_block->memory + base_block->capacity) {
      std::cerr << "address: " << address << " min: " << base_block->memory
                << " max: " << base_block->memory + base_block->capacity
                << std::endl;
      throw std::runtime_error("invalid allocation ptr out of range");
    }

    block_hash_[index] = address;
    base_memory_[(char*)address] = base_block;

    return address;
  }

  bool dealloc(Index3D index, BlockType* ptr) {
    // This block might be on the GPU, so we don't call the destructor on it,
    // but just make the block free.
    // ptr->~BlockType();
    auto it = base_memory_.find((char*)ptr);
    // This BlockType* hase not been allocated. Return to prevent double free.
    if (it == base_memory_.end()) {
      throw std::runtime_error("ptr has not been allocated");
    }

    // update the free block of *ptr to reflect the deallocated state
    MemoryBlock* mem_block = it->second;
    // assert(("Dealloc ptr must be in memory block range",
    //         (mem_block->memory <= ptr) &&
    //             (ptr <= mem_block->memory + mem_block->capacity)));
    if (mem_block->memory > ptr ||
        ptr > mem_block->memory + mem_block->capacity) {
      std::cerr << "ptr " << ptr << std::endl;
      std::cerr << "base_block" << mem_block->memory << std::endl;
      throw std::runtime_error("ptr is not in the base block memory range");
    }

    size_t block_idx = ptr - mem_block->memory;
    FreeBlock* new_free_block = &mem_block->free_ptrs[block_idx];
    if (new_free_block > mem_block->free_ptrs + mem_block->capacity) {
      throw std::runtime_error("new_free_block is not in block range");
    }
    new_free_block->next_free = reinterpret_cast<char*>(first_deleted_);
    new_free_block->base_block = reinterpret_cast<char*>(mem_block);
    first_deleted_ = new_free_block;

    block_hash_.erase(index);
    base_memory_.erase((char*)ptr);
    return true;
  }

  BlockType* isAllocated(Index3D index) {
    auto it = block_hash_.find(index);
    return it == block_hash_.end() ? nullptr : it->second;
  }

 private:
  struct FreeBlock {
    char* next_free = nullptr;
    char* base_block = nullptr;

    inline void reset() {
      next_free = nullptr;
      base_block = nullptr;
    }
  };

  struct MemoryBlock {
    BlockType* memory;
    size_t capacity;
    size_t count;
    FreeBlock* free_ptrs;

    int device;
    MemoryBlock* next_block;


    MemoryBlock(size_t capacity, int device, const cudaStream_t& stream)
        : count(0), capacity(capacity), device(device), next_block(nullptr) {
      if (capacity < 1) {
        throw std::invalid_argument("capacity has to be greater than 1");
      }
      memory =
          Allocator::allocate(sizeof(BlockType) * capacity, device, stream);
      free_ptrs = new FreeBlock[capacity];
      if (memory == nullptr) {
        throw std::bad_alloc();
      }
    }

    ~MemoryBlock() {
      Allocator::deallocate(memory);
      delete[] free_ptrs;
    }
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
      assert(("Allocated block size must be in the range [1, max_size]",
              (1 <= size) && (size <= max_block_length_)));
    }

    // create a new memory block and update the internal structure
    MemoryBlock* new_block = new MemoryBlock(size, device, stream);
    last_block_->next_block = new_block;
    last_block_ = new_block;
    block_memory_ = new_block->memory;
    count_in_block_ = 0;
    block_capacity_ = size;
  }

  // allocator management
  BlockType* block_memory_;
  FreeBlock* first_deleted_;
  size_t count_in_block_;
  size_t block_capacity_;
  MemoryBlock first_block_;
  MemoryBlock* last_block_;
  size_t max_block_length_;

  BlockHash block_hash_;
  std::unordered_map<char*, MemoryBlock*> base_memory_;

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
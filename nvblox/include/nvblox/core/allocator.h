// This allocator is based on
// https://www.codeproject.com/Articles/746630/O-Object-Pool-in-Cplusplus
#pragma once

#include <memory.h>
#include <algorithm>
#include <iomanip>
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
    checkCudaErrors(cudaMemsetAsync(ptr, 0, size, stream));
    checkCudaErrors(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation,
                                  cudaCpuDeviceId));
    // NOTE(rasaford): Create a mapping for the memory on the device, such that
    // it can access it directly
    checkCudaErrors(
        cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device));
    return ptr;
  }
  static inline void deallocate(T* ptr) { checkCudaErrors(cudaFree(ptr)); }
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
    checkCudaErrors(cudaMemsetAsync(ptr, 0, size, stream));
    checkCudaErrors(
        cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device));
    return ptr;
  }
  static inline void deallocate(T* ptr) { checkCudaErrors(cudaFree(ptr)); }
};

// Memory allocator for Device and host memory
// NOTE(rasaford): This pool will never free up memory, unless it is destroyed
template <typename BlockType, class Allocator>
class ObjectPool {
 private:
  struct Block {
    // Memory managed by this Block
    BlockType* memory;

    // Array of FreeBlock for each BlockType objects that could be allocated
    int* next_free;
    int free_idx;
    int count;
    size_t capacity;

    Block(size_t capacity, int device, const cudaStream_t& stream)
        : count(0), capacity(capacity), free_idx(-1) {
      if (capacity < 1) {
        throw std::invalid_argument("capacity has to be greater than 1");
      }
      memory =
          Allocator::allocate(sizeof(BlockType) * capacity, device, stream);
      if (memory == nullptr) {
        throw std::bad_alloc();
      }
      next_free = new int[capacity];
      std::fill_n(next_free, capacity, -1);
    }

    Block(const Block&) = delete;             // no-copy
    Block& operator=(const Block&) = delete;  // no copy-assignment

    // move constructor
    Block(Block&& block)
        // move the data to this block
        : memory(block.memory),
          next_free(block.next_free),
          free_idx(block.free_idx),
          count(block.count),
          capacity(block.capacity) {
      // delete owned ptrs from the other block
      block.memory = nullptr;
      block.next_free = nullptr;
    }
    // move-assignment
    Block& operator=(Block&& block) {
      // move the data to this block
      memory = block.memory;
      next_free = block.next_free;
      free_idx = block.free_idx;
      count = block.count;
      capacity = block.capacity;
      // delete owned ptrs from the other block
      block.memory = nullptr;
      block.next_free = nullptr;
      return *this;
    }

    ~Block() {
      if (memory != nullptr) Allocator::deallocate(memory);
      if (next_free != nullptr) delete[] next_free;
    }
  };

 public:
  ObjectPool(int device, cudaStream_t stream, size_t initial_capacity = 1 << 9,
             size_t max_block_length = 1 << 12)
      : block_size_(initial_capacity),
        max_block_length_(max_block_length),
        device(device),
        stream(stream) {
    if (max_block_length < 1) {
      throw std::invalid_argument("max_block_length must be at last 1");
    }
    blocks_.emplace_back(initial_capacity, device, stream);
    priorities_.emplace_back(0);
  };

  ~ObjectPool() {}

  BlockType* alloc() {
    BlockType* address = nullptr;
    int block_idx = priorities_[priorities_.size() - 1];
    for (const int b : priorities_) {
      if (blocks_[b].count < blocks_[b].capacity) {
        block_idx = b;
        break;
      }
    }
    Block& base_block = blocks_[block_idx];
    // std::cout << "block_idx " << block_idx << std::endl;

    if (base_block.free_idx != -1) {
      // reuse a previously freed block
      int chunk_idx = base_block.free_idx;
      base_block.free_idx = base_block.next_free[chunk_idx];
      base_block.next_free[chunk_idx] = -1;

      if (chunk_idx >= base_block.capacity) {
        throw std::runtime_error(
            "alloc: chunk_idx must be in range [0, capacity)");
      }

      address = base_block.memory + chunk_idx;
      cudaMemsetAsync(address, 0, sizeof(BlockType), stream);
    } else {
      // allocate a new block
      if (base_block.count >= base_block.capacity) {
        block_idx = allocate_block();
      }
      // if no chunks are free in a block, we can allocate by simply giving out
      // chunks in order
      address = blocks_[block_idx].memory + blocks_[block_idx].count;
    }

    if (address < blocks_[block_idx].memory ||
        address >= blocks_[block_idx].memory + blocks_[block_idx].capacity) {
      throw std::runtime_error("invalid allocation ptr out of range");
    }

    // store the allocated ptr for referencing in dealloc
    blocks_[block_idx].count++;
    allocated_[address] = block_idx;
    return address;
  }

  void dealloc(BlockType* pointer) {
    // NOTE(rasaford): This block might be on the GPU, so we don't call the
    // destructor on it, but just free the block.  ptr->~BlockType();
    const auto it = allocated_.find(pointer);
    // NOTE(rasaford): Check that this BlockType* has been allocated. Throw to
    // prevent double free.
    if (it == allocated_.end()) {
      throw std::runtime_error("dealloc: ptr has not been allocated");
    }

    // update the free block of *ptr to reflect the deallocated state
    const int block_idx = it->second;
    Block& block = blocks_[block_idx];
    if (block.memory > pointer || pointer > block.memory + block.capacity) {
      throw std::runtime_error(
          "dealloc: BlockType* is not in the base block memory range");
    }

    const size_t chunk_idx = pointer - block.memory;
    // Chunk* new_free_block = &block.free_ptrs[chunk_idx];

    if (chunk_idx >= block.capacity) {
      throw std::runtime_error(
          "dealloc: chunk_idx is not within the range [0, capacity)");
    }

    // update the Block
    block.next_free[chunk_idx] = block.free_idx;
    block.free_idx = chunk_idx;
    block.count--;

    if (block.count < 0) {
      std::runtime_error("invalid count");
    } else if (block.count == 0) {
      std::cout << "free memory block " << block_idx << std::endl;
      blocks_.erase(blocks_.begin() + block_idx);
      auto prio = std::find(priorities_.begin(), priorities_.end(), block_idx);
      priorities_.erase(prio);
    }

    // free block tracking based on minimal free space
    const int free_count = block.capacity - block.count;
    if (free_count < min_free_count_) {
      // insert in priorities (similar to insertion sort)
      const int block_prio =
          std::find(priorities_.begin(), priorities_.end(), block_idx) -
          priorities_.begin();
      for (int i = block_prio; i > 0; i--) priorities_[i] = priorities_[i - 1];
      priorities_[0] = block_idx;

      min_free_count_ = free_count;
      // // new_free_block->next_free = next_free_chunk_
      // next_free_chunk_ = new_free_block;
      // next_free_idx_ = block_idx;
    }

    // first_deleted_ = new_free_block;

    allocated_.erase(pointer);
  }

  bool isAllocated(BlockType* pointer) const {
    return (pointer != nullptr) &&
           (allocated_.find(pointer) != allocated_.end());
  }

  void printUsage() const {
    int min = INT_MAX;
    int max = INT_MIN;
    int sum = 0;
    int num_blocks = 0;
    int total_count = 0;
    int total_cap = 0;
    for (const Block& block : blocks_) {
      min = std::min(min, block.count);
      max = std::max(max, block.count);
      sum += block.count;
      total_count += block.count;
      total_cap += block.capacity;
      num_blocks++;
    }
    std::cout << "Blocks Usage: " << std::setw(6) << total_count << " / "
              << std::setw(6) << total_cap << " "
              << 100.f * (float)total_count / total_cap << "%"
              << " Num Blocks: " << std::setw(3) << num_blocks
              << " Memory Usage: " << std::setw(5)
              << total_count * sizeof(BlockType) / (1024 * 1024) << " / "
              << std::setw(5) << total_cap * sizeof(BlockType) / (1024 * 1024)
              << " MiB" << std::endl;
  }

 protected:
  int allocate_block() {
    // size_t size = block_size_;
    // if (size < max_block_length_) {
    //   size *= 2;
    //   // if (size < 0) {
    //   //   std::overflow_error(
    //   //       "allocation is too big, resulting in block_size_ overflow");
    //   // }
    // }
    block_size_ = std::min(block_size_ * 2, max_block_length_);

    // create a new memory block and update the internal management structure
    const int block_idx = blocks_.size();
    std::cout << "allocate new block " << block_size_ << " " << device << " "
              << stream << std::endl;
    blocks_.emplace_back(block_size_, device, stream);
    priorities_.emplace_back(block_idx);
    return block_idx;
  }

  // allocator management
  std::vector<Block> blocks_;
  std::vector<int> priorities_;

  int next_free_idx_;
  size_t min_free_count_;

  size_t block_size_;
  size_t max_block_length_;
  // allocated index keeping track of live allocations
  std::unordered_map<BlockType*, int> allocated_;
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
    cudaStreamCreate(&host_stream);
    cudaStreamCreate(&device_stream);
    host_pool = std::make_unique<HostPool>(device, host_stream);
    device_pool = std::make_unique<DevicePool>(device, device_stream);
  };

  ~Allocator() {
    cudaStreamDestroy(host_stream);
    cudaStreamDestroy(device_stream);
  }

  typename BlockType::Ptr toDevice(typename BlockType::Ptr block) {
    timing::Timer to_device_timer("prefetch/prefetch/to_device");
    BlockType* ptr = block.get();
    if (device_pool->isAllocated(ptr)) {
      return block;
    }

    BlockType* device_ptr = device_pool->alloc();
    if (ptr != nullptr && host_pool->isAllocated(ptr)) {
      cudaMemcpyAsync(device_ptr, ptr, sizeof(BlockType), cudaMemcpyDefault,
                      device_stream);
      host_pool->dealloc(ptr);
    }
    return unified_ptr<BlockType>(device_ptr, MemoryType::kPool);
  }

  typename BlockType::Ptr toHost(typename BlockType::Ptr block) {
    timing::Timer to_host_timer("prefetch/evict/to_host");
    BlockType* ptr = block.get();
    if (host_pool->isAllocated(ptr)) {
      return block;
    }

    BlockType* host_ptr = host_pool->alloc();
    if (ptr != nullptr && device_pool->isAllocated(ptr)) {
      cudaMemcpyAsync(host_ptr, ptr, sizeof(BlockType), cudaMemcpyDefault,
                      host_stream);
      device_pool->dealloc(ptr);
    }
    return unified_ptr<BlockType>(host_ptr, MemoryType::kPool);
  }

  void printUsage() {
    std::cout << typeid(BlockType).name() << " DevicePool ";
    device_pool->printUsage();
    std::cout << typeid(BlockType).name() << " HostPool   ";
    host_pool->printUsage();
  }

  void waitForAllocations() {
    checkCudaErrors(cudaStreamSynchronize(host_stream));
    checkCudaErrors(cudaStreamSynchronize(device_stream));
  }

 protected:
  std::unique_ptr<HostPool> host_pool;
  std::unique_ptr<DevicePool> device_pool;

  cudaStream_t host_stream, device_stream;
  int device;
};

}  // namespace alloc
}  // namespace nvblox
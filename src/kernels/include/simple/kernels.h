#pragma once

#include <hipWrapper.h>

namespace gpuio::kernels::simple {

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD, unsigned WARP_SIZE = 64>
__device__ __forceinline__ T BlockSum(
    T  item,
    T* shared
    ) {
  __syncthreads();

  T val = item;
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  // Calculate sum across warp
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    // val += __shfl_down_sync(0xffffffffffffffff, val, offset);
    val += __shfl_down(val, offset); // on AMD GPUs warps are always lock-step
  }

  // Store sum in buffer
  if (lane == 0) {
    shared[wid] = val;
  }

  __syncthreads();

  // Load the sums into the first warp
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

  // Calculate sum of sums
  if (wid == 0) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      // val += __shfl_down_sync(0xffffffffffffffff, val, offset);
      val += __shfl_down(val, offset); // on AMD GPUs warps are always lock-step
    }
  }

  return val;
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ T BlockSum(
    T (&items)[ITEMS_PER_THREAD],
    T* shared
    ) {
  T thread_sum = 0;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    thread_sum += items[ITEM];
  }

  return BlockSum(thread_sum, shared);
} 

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirect(
    const unsigned int tid,
    T* block_itr,
    T  (&items)[ITEMS_PER_THREAD]
    ) {
  T* thread_itr = block_itr + tid;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
  }
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirect(
    const unsigned int tid,
    T* block_itr,
    T  (&items)[ITEMS_PER_THREAD],
    int num_items
    ) {
  T* thread_itr = block_itr + tid;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
    }
  }
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoad(
    T* inp,
    T  (&items)[ITEMS_PER_THREAD],
    int num_items
    ) {
  T* block_itr = inp;

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockLoadDirect<T, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, block_itr, items);
  } else {
    BlockLoadDirect<T, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, block_itr, items, num_items);
  }
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirectCond(
    const unsigned int tid,
    T* block_itr,
    T  (&items)[ITEMS_PER_THREAD],
    int (&flags)[ITEMS_PER_THREAD]
    ) {
  T* thread_itr = block_itr + tid;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (flags[ITEM]) items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
  }
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirectCond(
    const unsigned int tid,
    T* block_itr,
    T  (&items)[ITEMS_PER_THREAD],
    int (&flags)[ITEMS_PER_THREAD],
    int num_items
    ) {
  T* thread_itr = block_itr + tid;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (flags[ITEM]) items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
    }
  }
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadCond(
    T* inp,
    T  (&items)[ITEMS_PER_THREAD],
    int (&flags)[ITEMS_PER_THREAD],
    int num_items
    ) {
  T* block_itr = inp;

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockLoadDirectCond<T, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, block_itr, items, flags);
  } else {
    BlockLoadDirectCond<T, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, block_itr, items, flags, num_items);
  }
}


template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__
void sumSelectiveKernel(int *pred, T *vals, int N, uint64_t *out) {
  int selection_flags[ITEMS_PER_THREAD];
  T v[ITEMS_PER_THREAD];
  constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = N - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(pred + tile_offset, selection_flags, num_tile_items);
  BlockLoadCond<T, BLOCK_THREADS, ITEMS_PER_THREAD>(vals + tile_offset, v, selection_flags, num_tile_items);

  uint64_t sum = 0;
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      if (selection_flags[ITEM])
        sum += v[ITEM];
  }

  __syncthreads();

  using result_t = uint64_t;
  static __shared__ result_t buffer[32];
  result_t aggregate = BlockSum<result_t, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (result_t *)buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(out, aggregate);
  }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
void sumSelective(
  int *pred, T *vals, int N, uint64_t *out, hipStream_t s
) {
  int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;

  gpuio::hip::LanuchKernel(sumSelectiveKernel<T, BLOCK_THREADS, ITEMS_PER_THREAD>, 
    dim3((N + tile_items - 1) / tile_items), dim3(BLOCK_THREADS), 0, s,
    pred, vals, N, out
  );
}

} // namespace gpuio::kernels::simple
#pragma once
#include <hipWrapper.h>

namespace crystal {

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

} // namespace crystal
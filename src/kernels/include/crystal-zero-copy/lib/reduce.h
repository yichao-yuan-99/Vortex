#pragma once
#include <hipWrapper.h>

namespace crystal {

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
} // namespace crystal

}
#pragma once

#include <crystal/lib/crystal.h>

namespace crystal::q1x {

using result_t = unsigned long long;

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ 
void Q11Kernel(int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice,
    int lo_num_entries, result_t* revenue) {
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  result_t sum = 0;

  constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_num_entries + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = lo_num_entries - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset, items, num_tile_items);
  BlockPredGT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19930000, selection_flags, num_tile_items);
  BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19940000, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_quantity + tile_offset, items, num_tile_items);
  BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 25, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_discount + tile_offset, items, num_tile_items);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 3, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_extendedprice + tile_offset, items2, num_tile_items);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      if (selection_flags[ITEM])
        sum += items[ITEM] * items2[ITEM];
  }

  __syncthreads();

  static __shared__ result_t buffer[32];
  result_t aggregate = BlockSum<result_t, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(revenue, aggregate);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
void Q11Run(
  int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice,
  int lo_len, result_t* revenue, hipStream_t s) {
  int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;

  gpuio::hip::LanuchKernel(Q11Kernel<BLOCK_THREADS, ITEMS_PER_THREAD>,
    dim3((lo_len + tile_items - 1) / tile_items), dim3(BLOCK_THREADS), 0, s,
    lo_orderdate, lo_discount, lo_quantity, lo_extendedprice, lo_len, revenue
  );
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ 
void Q12Kernel(int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice,
    int lo_num_entries, result_t* revenue) {
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  result_t sum = 0;
  
  constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_num_entries + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = lo_num_entries - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset, items, num_tile_items);
  BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19940101, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19940131, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_quantity + tile_offset, items, num_tile_items);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 26, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 35, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_discount + tile_offset, items, num_tile_items);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 4, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 6, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_extendedprice + tile_offset, items2, num_tile_items);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items)
      if (selection_flags[ITEM])
        sum += items[ITEM] * items2[ITEM];
  }

  __syncthreads();

  static __shared__ result_t buffer[32];
  result_t aggregate = BlockSum<result_t, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (result_t *)buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(revenue, aggregate);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
void Q12Run(
  int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice,
  int lo_len, result_t* revenue, hipStream_t s) {
  int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;

  gpuio::hip::LanuchKernel(Q12Kernel<BLOCK_THREADS, ITEMS_PER_THREAD>,
    dim3((lo_len + tile_items - 1) / tile_items), dim3(BLOCK_THREADS), 0, s,
    lo_orderdate, lo_discount, lo_quantity, lo_extendedprice, lo_len, revenue
  );
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ 
void Q13Kernel(int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice,
    int lo_num_entries, result_t* revenue) {
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  result_t sum = 0;

  constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_num_entries + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = lo_num_entries - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset, items, num_tile_items);
  BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19940204, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19940210, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_quantity + tile_offset, items, num_tile_items);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 26, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 35, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_discount + tile_offset, items, num_tile_items);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 5, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 7, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_extendedprice + tile_offset, items2, num_tile_items);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      if (selection_flags[ITEM])
        sum += items[ITEM] * items2[ITEM];
  }

  __syncthreads();

  static __shared__ result_t buffer[32];
  result_t aggregate = BlockSum<result_t, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (result_t *)buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(revenue, aggregate);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
void Q13Run(
  int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice,
  int lo_len, result_t* revenue, hipStream_t s) {
  int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;

  gpuio::hip::LanuchKernel(Q13Kernel<BLOCK_THREADS, ITEMS_PER_THREAD>,
    dim3((lo_len + tile_items - 1) / tile_items), dim3(BLOCK_THREADS), 0, s,
    lo_orderdate, lo_discount, lo_quantity, lo_extendedprice, lo_len, revenue
  );
}




} // namespace crystal::q1x
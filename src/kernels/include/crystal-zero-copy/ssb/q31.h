#pragma once

#include <crystal-zero-copy/lib/crystal.h>

namespace crystal::q31 {

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe(int* lo_orderdate, int* lo_custkey, int* lo_suppkey, int* lo_revenue, int lo_len,
    int* ht_s, int s_len,
    int* ht_c, int c_len,
    int* ht_d, int d_len,
    int* res) {
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int c_nation[ITEMS_PER_THREAD];
  int s_nation[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = lo_len - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoadCond<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_custkey + tile_offset, items, selection_flags, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, c_nation, selection_flags,
      ht_c, c_len, num_tile_items);

  BlockLoadCond<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset, items, selection_flags, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, s_nation, selection_flags,
      ht_s, s_len, num_tile_items);

  BlockLoadCond<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset, items, selection_flags, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, year, selection_flags,
      ht_d, d_len, 19920101, num_tile_items);

  BlockLoadCond<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_revenue + tile_offset, revenue, selection_flags, num_tile_items);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) {
      if (selection_flags[ITEM]) {
        int hash = (s_nation[ITEM] * 25 * 7  + c_nation[ITEM] * 7 +  (year[ITEM] - 1992)) % ((1998-1992+1) * 25 * 25);
        res[hash * 6] = year[ITEM];
        res[hash * 6 + 1] = c_nation[ITEM];
        res[hash * 6 + 2] = s_nation[ITEM];
        /*atomicAdd(&res[hash * 6 + 4], revenue[ITEM]);*/
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(revenue[ITEM]));
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__
void build_hashtable_s(int *filter_col, int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 2, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, items2, selection_flags, 
      hash_table, num_slots, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__
void build_hashtable_c(int *filter_col, int *dim_key, int* dim_val, int num_tuples, int *hash_table, int num_slots) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 2, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, items2, selection_flags, 
      hash_table, num_slots, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__
void build_hashtable_d(int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items, num_tile_items);
  BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1992, selection_flags, num_tile_items);
  BlockPredLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1997, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, items, selection_flags, 
      hash_table, num_slots, 19920101, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
void buildTables(
  int *d_datekey, int* d_year, int d_len,
  int *s_suppkey, int* s_region, int* s_nation, int s_len,
  int *c_custkey, int* c_region, int* c_nation, int c_len,
  int *ht_d, int *ht_s, int *ht_c,
  hipStream_t s
) {
  int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;

  gpuio::hip::LanuchKernel(build_hashtable_s<BLOCK_THREADS, ITEMS_PER_THREAD>, 
    dim3((s_len + tile_items - 1) / tile_items), dim3(BLOCK_THREADS), 0, s,
    s_region, s_suppkey, s_nation, s_len, ht_s, s_len
  );

  gpuio::hip::LanuchKernel(build_hashtable_c<BLOCK_THREADS, ITEMS_PER_THREAD>, 
    dim3((c_len + tile_items - 1) / tile_items), dim3(BLOCK_THREADS), 0, s,
    c_region, c_custkey, c_nation, c_len, ht_c, c_len
  );

  int d_val_min = 19920101;
  int d_val_len = 19981230 - 19920101 + 1;
  gpuio::hip::LanuchKernel(build_hashtable_d<BLOCK_THREADS, ITEMS_PER_THREAD>, 
    dim3((d_len + tile_items - 1) / tile_items), dim3(BLOCK_THREADS), 0, s,
    d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min
  );
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
void probleTables(
  int* lo_orderdate, int* lo_custkey, int* lo_suppkey, int* lo_revenue, int lo_len,
  int *ht_d, int *ht_s, int *ht_c,
  int d_len, int s_len, int c_len,
  int *res,
  hipStream_t s
) {
  int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
  int d_val_len = 19981230 - 19920101 + 1;

  gpuio::hip::LanuchKernel(probe<BLOCK_THREADS, ITEMS_PER_THREAD>,
    dim3((lo_len + tile_items - 1) / tile_items), dim3(BLOCK_THREADS), 0, s,
     lo_orderdate, lo_custkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len, ht_c, c_len, ht_d, d_val_len, res
  );
}

} // namespace crystal::q31
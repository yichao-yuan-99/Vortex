#pragma once

#include <crystal-zero-copy/lib/crystal.h>

namespace crystal::q2x {

template<int BLOCK_THREADS, int ITEMS_PER_THREAD, int V>
__global__ 
void build_hashtable_s(int *filter_col, int *dim_key, int num_tuples, int *hash_table, int num_slots) {
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, V, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, 
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

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, items2, selection_flags, 
      hash_table, num_slots, val_min, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ 
void build_hashtable_p_q21(int *filter_col, int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots) {
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
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, items2, selection_flags, 
      hash_table, num_slots, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_p_q22(int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots) {
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
  BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 260, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 267, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, items, selection_flags, 
      hash_table, num_slots, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ 
void build_hashtable_p_q23(int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots) {
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
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 260, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, items, selection_flags, 
      hash_table, num_slots, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe(int* lo_orderdate, int* lo_partkey, int* lo_suppkey, int* lo_revenue, int lo_len,
    int* ht_s, int s_len,
    int* ht_p, int p_len,
    int* ht_d, int d_len,
    int* res) {
  // Load a tile striped across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int brand[ITEMS_PER_THREAD];
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

  BlockLoadCond<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_partkey + tile_offset, items, selection_flags, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, brand, selection_flags,
      ht_p, p_len, num_tile_items);

  BlockLoadCond<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset, items, selection_flags, num_tile_items);
  BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, ht_s, s_len, num_tile_items);


  BlockLoadCond<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset, items, selection_flags, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, year, selection_flags,
      ht_d, d_len, 19920101, num_tile_items);

  BlockLoadCond<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_revenue + tile_offset, revenue, selection_flags, num_tile_items);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) {
      if (selection_flags[ITEM]) {
        int hash = (brand[ITEM] * 7 +  (year[ITEM] - 1992)) % ((1998-1992+1) * (5*5*40));
        res[hash * 4] = year[ITEM];
        res[hash * 4 + 1] = brand[ITEM];
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]), (long long)(revenue[ITEM]));
      }
    }
  }
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD, int QUERY>
void buildTables(
  int* p_partkey, int* p_brand1, int *p_category, int p_len,
  int *d_datekey, int* d_year, int d_len,
  int *s_suppkey, int* s_region, int s_len,
  int *ht_d, int *ht_p, int *ht_s,
  hipStream_t s
) {
  int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;

  gpuio::hip::LanuchKernel(build_hashtable_s<BLOCK_THREADS, ITEMS_PER_THREAD, QUERY>, 
    dim3((s_len + tile_items - 1) / tile_items), dim3(BLOCK_THREADS), 0, s,
    s_region, s_suppkey, s_len, ht_s, s_len
  );

  if constexpr(QUERY == 1) {
    gpuio::hip::LanuchKernel(build_hashtable_p_q21<BLOCK_THREADS, ITEMS_PER_THREAD>,
      dim3((p_len + tile_items - 1) / tile_items), dim3(BLOCK_THREADS), 0, s,
      p_category, p_partkey, p_brand1, p_len, ht_p, p_len
    );
  } else if constexpr(QUERY == 2) {
    gpuio::hip::LanuchKernel(build_hashtable_p_q22<BLOCK_THREADS, ITEMS_PER_THREAD>,
      dim3((p_len + tile_items - 1) / tile_items), dim3(BLOCK_THREADS), 0, s,
      p_partkey, p_brand1, p_len, ht_p, p_len
    );
  } else if constexpr(QUERY == 3) {
    gpuio::hip::LanuchKernel(build_hashtable_p_q23<BLOCK_THREADS, ITEMS_PER_THREAD>,
      dim3((p_len + tile_items - 1) / tile_items), dim3(BLOCK_THREADS), 0, s,
      p_partkey, p_brand1, p_len, ht_p, p_len
    );
  } else {
    static_assert(0);
  }

  int d_val_min = 19920101;
  int d_val_len = 19981230 - 19920101 + 1;
  gpuio::hip::LanuchKernel(build_hashtable_d<BLOCK_THREADS, ITEMS_PER_THREAD>,
    dim3((d_len + tile_items - 1) / tile_items), dim3(BLOCK_THREADS), 0, s,
    d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min
  );
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
void probleTables(int* lo_orderdate, int* lo_partkey, int* lo_suppkey, int* lo_revenue, int lo_len,
    int* ht_s, int s_len,
    int* ht_p, int p_len,
    int* ht_d, int d_len,
    int* res, hipStream_t s
) {
  int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
  int d_val_len = 19981230 - 19920101 + 1;

  gpuio::hip::LanuchKernel(probe<BLOCK_THREADS, ITEMS_PER_THREAD>,
    dim3((lo_len + tile_items - 1) / tile_items), dim3(BLOCK_THREADS), 0, s,
    lo_orderdate, lo_partkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len, ht_p, p_len, ht_d, d_val_len, res
  );
}

} // namespace crystal::q2x
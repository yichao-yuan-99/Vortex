#pragma once

#include <hipWrapper.h>

// kernels here assumes a stack-based SIMT scheduler (pre-volta and on AMD CDNA)
// this is becasue hip do not provide warp sync primitives

namespace gpuio::zcpy {

template <size_t N = 1000'000'000, unsigned id = 0>
__global__
void pingpong(volatile int *A, volatile int *B) {
  unsigned wuid = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
  A += wuid;
  B += wuid;

  if (threadIdx.x % 64 == 0) {
    for (size_t i = 0; i < N; i++) {
      while(*A != -1) continue;
      *A = 0;
      *B = -1;
      __threadfence_system();
      printf("[%u, %u] %d\n", id, wuid, (int) i);
    }
    *B = -1;
    __threadfence_system();
    printf("[%u, %u] done\n", id, wuid);
  }


}

/*
 * run on the forwarding device, and need to make sure that each SM are concurrently running maximum
 * number of threads.
 * On MI100, this means 2560 threads, 2560 / 64 = 40 warps.
 */
template<typename T, unsigned BLOCK_SIZE = 512, unsigned WARP_SIZE = 64>
__global__
void WarpForwardingLoop(volatile uint64_t *mask, T *volatile *base_p, volatile T *vals, volatile unsigned *comp) {
  unsigned wuid = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + threadIdx.x / WARP_SIZE;

  mask += wuid;
  base_p += wuid;
  vals += wuid * WARP_SIZE;
  comp += wuid;

  // To end the loop, the thread need to observe the following sequence
  // 1. base is writte as -1 and 2. mask is written as non-0
  if (threadIdx.x % WARP_SIZE == 0) {
    while(1) {
      uint64_t m = 0;
      while((m = *mask) == 0) continue;

      T *base = *base_p;
      if (base == reinterpret_cast<T *>(0xffffffffffffffff)) break;
      // if (threadIdx.x % WARP_SIZE == 0) {
        // printf("recv\n");
      // }
      // if (m & (1 << (threadIdx.x % WARP_SIZE))) {
      //   vals[threadIdx.x % WARP_SIZE] = base[threadIdx.x % WARP_SIZE];
      // }

      // __threadfence_system();
      *mask = 0;
      // __threadfence_system();
      *comp = 1;
    }
  }
}


template <unsigned NUM_CU = 120, unsigned MAX_NUM_WARP = 40, unsigned WARP_SIZE = 64>
constexpr inline unsigned valBlockCnt() { return NUM_CU * MAX_NUM_WARP * WARP_SIZE; }

template <unsigned NUM_CU = 120, unsigned MAX_NUM_WARP = 40>
constexpr inline unsigned ctrlBlockCnt() { return NUM_CU * MAX_NUM_WARP; }

template <unsigned BLOCK_SIZE = 512, unsigned NUM_CU = 120, unsigned MAX_THREAD_CU = 2560>
constexpr inline unsigned lanuchLoopingBlockNum() { return NUM_CU * (MAX_THREAD_CU / BLOCK_SIZE); }

/*
 * run on the forwarding device
 * the request will be sent to the warp on the remote GPU
 */
template<typename T, unsigned WARP_SIZE = 64>
__device__
void WarpForwardingGet(volatile uint64_t *mask, T *volatile *base_p, volatile T *vals, 
  volatile unsigned *locks, volatile unsigned *comp,
  unsigned slot, int cond, T *base, T &v) {
  // offset the addresses to another gpu
  locks += slot;
  mask += slot;
  comp += slot;
  base_p += slot;
  vals += slot * WARP_SIZE;

  if (threadIdx.x % WARP_SIZE == 0) {
    T old;
    do {
      old = atomicCAS(const_cast<unsigned *>(locks), 0, 1);
    } while (old != 0);
  }


  // if (threadIdx.x % WARP_SIZE == 0) {
  //   *base_p = base;
  // }

  // __threadfence_system();

  uint64_t m = __ballot(cond);
  if (threadIdx.x % WARP_SIZE == 0) {
    *mask = m;
    *comp = 0;
    // __threadfence_system();
    printf("send %d %d\n", (int) blockIdx.x, (int) (threadIdx.x / WARP_SIZE));
    while (*comp == 0) continue;
  }


  // if (cond) v = vals[threadIdx.x % WARP_SIZE];

  if (threadIdx.x % WARP_SIZE == 0) {
    *locks = 0;
    __threadfence();
  }

  // if (threadIdx.x % WARP_SIZE == 0) {
  //   *base_p = base;
  // }

  // __threadfence_system();

  // if (threadIdx.x % WARP_SIZE == 0) {
  //   *mask = m;
  // }

  // while (*mask) continue;

  // if (cond) v = vals[threadIdx.x % WARP_SIZE];
}

// note that the value returned by smid is not sequential.
// the last 3 bit is cu id in each shader array
// on MI100, each se array has 15 CUs
// template <typename T = int>
// __device__ __forceinline__
// T smidSeq() {
//   int smid = __smid();
//   return smid - (smid >> 3);
// }

// A set of functions to get a unique id for each block in a CU
// 
// if block size is 256, then each CU can concurrently run at most 10 blocks on MI100
// each CU owns a 32 entry ring buffer, initialized with [0, 1, ... 9, -1, -1, ... -1]
// The fifo is used as a free list
// template <unsigned NUM_ENTRY = 32>
// __device__ __forceinline__
// unsigned fifoGet(unsigned *head, volatile unsigned *vals) {
//   unsigned pos = atomicInc(head, NUM_ENTRY);
//   return vals[pos];
// }

// template <unsigned NUM_ENTRY = 32>
// __device__ __forceinline__
// void fifoPut(unsigned *tail, volatile unsigned *vals, unsigned buid) {
//   unsigned pos = atomicInc(tail, NUM_ENTRY);
//   vals[pos] = buid;
// }

// template <unsigned NUM_ENTRY = 32>
// __device__ __forceinline__
// unsigned getBlockUidinCU(volatile unsigned *ctrls) {
//   int cuid = smidSeq();
//   assert(cuid < 120);
//   ctrls += cuid * (NUM_ENTRY + 2);
//   auto r = fifoGet<NUM_ENTRY>(const_cast<unsigned *>(ctrls), ctrls + 2);
//   // printf("get %d %u\n", cuid, r);
//   return r;
// }

// template <unsigned NUM_ENTRY = 32>
// __device__ __forceinline__
// void freeBlockUidsinCU(volatile unsigned *ctrls, unsigned buid) {
//   int cuid = smidSeq();
//   assert(cuid < 120);
//   ctrls += cuid * (NUM_ENTRY + 2);
//   // printf("put %d %u\n", cuid, buid);
//   fifoPut<NUM_ENTRY>(const_cast<unsigned *>(ctrls + 1), ctrls + 2, buid);
// }

// generate warp id based on block id
// template <unsigned BLOCK_SIZE = 256, unsigned NUM_ENTRY = 32, unsigned WARP_SIZE = 64>
// __device__ __forceinline__
// unsigned getWarpUidinCU(volatile unsigned *ctrls, unsigned *s_buf) {
//   constexpr unsigned WARP_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
//   if (threadIdx.x == 0) {
//     *s_buf = getBlockUidinCU<NUM_ENTRY>(ctrls);
//     assert(*s_buf < 10);
//   }
//   __syncthreads();
//   assert(*s_buf < 10);

//   return (*s_buf) * WARP_PER_BLOCK + (threadIdx.x / WARP_SIZE);
// }

// template <unsigned BLOCK_SIZE = 256, unsigned NUM_ENTRY = 64, unsigned WARP_SIZE = 64>
// __device__ __forceinline__
// void freeWarpUidinCU(volatile unsigned *ctrls, unsigned wuid) {
//   constexpr unsigned WARP_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
//   if (threadIdx.x == 0) {
//     freeBlockUidsinCU<NUM_ENTRY>(ctrls, wuid / WARP_PER_BLOCK);
//   }
// }

// initialize the free list of block ids for each CU
// Assume 256 threads per block, at most 10 blocks
template <unsigned BLOCK_SIZE = 256, unsigned NUM_ENTRY = 32, unsigned MAX_THREAD_CU = 2560, unsigned NUM_CU = 120>
void fillInitialFreeListHost(gpuio::hip::HostVector<unsigned> &vals) {
  constexpr unsigned CHUNK_SIZE = NUM_ENTRY + 2;
  constexpr unsigned BLOCK_NUM = MAX_THREAD_CU / BLOCK_SIZE;
  assert(vals.size() == NUM_CU * CHUNK_SIZE);

  for (unsigned i = 0; i < NUM_CU; i++) {
    unsigned offset = i * CHUNK_SIZE;
    vals[offset] = 0;
    vals[offset + 1] = BLOCK_NUM - 1;
    for (unsigned j = 0; j < BLOCK_NUM; j++) {
      vals[offset + 2 + j] = j;
    }
    for (unsigned j = BLOCK_NUM; j < NUM_ENTRY; j++) {
      vals[offset + 2 + j] = 0xffffffff;
    }
  }
}

template <unsigned NUM_ENTRY = 32, unsigned NUM_CU = 120>
constexpr inline unsigned freeListBlockCnt() { return NUM_CU * (NUM_ENTRY + 2); }
/*
 * test kernels
 */
template <typename T>
__global__
void sumSelect(
  volatile uint64_t *mask, T *volatile *base_p, volatile T *vals, volatile unsigned *locks, volatile unsigned *comp,
  T *src, uint64_t *acc
) {
  // __shared__ unsigned s_buf;
  // unsigned wuid = getWarpUidinCU(ctrls, &s_buf);
  unsigned slot = (blockIdx.x * blockDim.x + threadIdx.x) % 480;

  T v;
  WarpForwardingGet(mask, base_p, vals, locks, comp, slot, 1, src + threadIdx.x, v);

  atomicAdd(acc, static_cast<uint64_t>(v));

  // freeWarpUidinCU(ctrls, wuid);
  // __threadfence();
}


} // namespace gpuio::zcpy

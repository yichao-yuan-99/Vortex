#pragma once

#include <sort/sort.h>

namespace gpuio::kernels::join {

template <typename T>
constexpr T rangeMask(unsigned begin_bit, unsigned end_bit) {
  return ((static_cast<T>(1) << end_bit) - 1) ^ ((static_cast<T>(1) << begin_bit) - 1); 
}

constexpr size_t rangeSize(unsigned begin_bit, unsigned end_bit) {
  return static_cast<size_t>(1) << (end_bit - begin_bit);
}

constexpr size_t rangeDataSize(unsigned begin_bit, unsigned end_bit) {
  return (rangeSize(begin_bit, end_bit) + 1) * sizeof(int);
}

template <typename T>
constexpr size_t byteSize(size_t count) {
  return count * sizeof(T);
}

template <typename K, typename V>
constexpr size_t radixParitionOutBufferSize(size_t size, unsigned begin_bit, unsigned end_bit) {
  return byteSize<K>(size) + byteSize<V>(size) + rangeDataSize(begin_bit, end_bit);
}

template <typename K, typename V>
constexpr size_t radixParitionInBufferSize(size_t size, unsigned, unsigned) {
  return byteSize<K>(size) + byteSize<V>(size);
}

template <typename T>
void radix_partition(
  void *temp_ptr, size_t &temp_size, double_buffer &key, double_buffer &val, MemoryRef ranges, 
  size_t size, unsigned begin_bit, unsigned end_bit, hipStream_t s = 0, bool check = false
) {
  gpuio::kernels::sort::rocm::radix_sort_pairs<T>(
    temp_ptr, temp_size, key, val, size, begin_bit, end_bit, s, check
  );
  if (temp_ptr) {
    gpuio::hip::MemsetAsync(ranges, 0xff, s);
    gpuio::kernels::sort::findBoundary<T>(key.current(), size, ranges, begin_bit, end_bit, s);
    gpuio::kernels::sort::patchBoundary(ranges, rangeSize(begin_bit, end_bit), s);
  }
}


/*
 * Kernels for hash join phase
 */
// https://github.com/JonathanWatkins/CUDA/blob/master/NvidiaCourse/Exercises/transpose/transpose.cu
// set BLOCKDIM to the number of partitions
template <typename T, unsigned BLOCKDIM>
__global__ 
void transposeKernel(T *idata, T *odata, int width, int height)
{
	__shared__ T block[BLOCKDIM][BLOCKDIM+1];
	
	unsigned int xIndex = blockIdx.x * BLOCKDIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCKDIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];

	}

	__syncthreads();

	xIndex = blockIdx.y * BLOCKDIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCKDIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

template <typename T, unsigned BLOCKDIM = 8>
void transpose(T *mIn, T *mOut, size_t sizeX, size_t sizeY, hipStream_t s) {
  size_t gx = (sizeX + BLOCKDIM - 1) / BLOCKDIM;
  size_t gy = (sizeY + BLOCKDIM - 1) / BLOCKDIM;
  gpuio::hip::LanuchKernel(transposeKernel<T, BLOCKDIM>, dim3(gx, gy), dim3(BLOCKDIM, BLOCKDIM), 
    0, s, mIn, mOut, sizeX, sizeY
  );
};

template <typename T>
__global__
void removeFoRKernel(T *mIn, T *mOut, size_t width, size_t height) {
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int indexD = yIndex * width + xIndex;
    unsigned int indexFoR = xIndex;
    mOut[indexD] = mIn[indexD] - mIn[indexFoR];
	}
}

template <typename T>
__global__
void smallPrefixSumKernel(T *in, T *out, size_t N) {
  unsigned tid = threadIdx.x;
  if (tid < N && tid > 0) {
    T s = 0;
    for (size_t i = 0; i < tid; i++) {
      s += in[i];
    }
    out[tid] = s;
  }
}

template <typename T>
__global__ 
void resetFoRKernel(T *m, size_t width, size_t height) {
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if((xIndex < width) && (yIndex < height) && (yIndex > 0))
	{
		unsigned int indexD = yIndex * width + xIndex;
    unsigned int indexFoR = xIndex;
    m[indexD] += m[indexFoR];
	}

}

template <typename T, unsigned BLOCKDIMX = 8, unsigned BLOCKDIMY = 32>
void removeFoR(T *mIn, T *mOut, size_t sizeX, size_t sizeY, hipStream_t s) {
  size_t gx = (sizeX + BLOCKDIMX - 1) / BLOCKDIMX;
  size_t gy = (sizeY + BLOCKDIMY - 1) / BLOCKDIMY;
  gpuio::hip::LanuchKernel(removeFoRKernel<T>, dim3(gx, gy), dim3(BLOCKDIMX, BLOCKDIMY), 0, s,
    mIn, mOut, sizeX, sizeY
  );
  gpuio::hip::LanuchKernel(smallPrefixSumKernel<T>, dim3(1), dim3(sizeX), 0, s,
    mOut + (sizeY - 1) * sizeX, mOut, sizeX
  );
  gpuio::hip::LanuchKernel(resetFoRKernel<T>, dim3(gx, gy), dim3(BLOCKDIMX, BLOCKDIMY), 0, s,
    mOut, sizeX, sizeY
  );
}

inline void regulateRanges(void *temp_ptr, size_t &temp_size, 
  int *mIn, int *mOut, size_t sizeX, size_t sizeY, hipStream_t s, bool check = false) {
  if (temp_ptr == nullptr) {
    temp_size = sizeX * sizeY * sizeof(int);
    return;
  }
  
  if (check) {
    if (temp_size < sizeX * sizeY * sizeof(int)) throw std::runtime_error("not enough temp space to regulate ranges");
  }

  int *tempD = reinterpret_cast<int *>(temp_ptr);
  transpose(mIn, tempD, sizeX, sizeY, s);
  removeFoR(tempD, mOut, sizeY, sizeX, s);
}

template <typename T, unsigned SHIFT = 24>
struct ShiftHash{
  __device__ __host__ __forceinline__ 
  T operator()(T k) {
    return k >> SHIFT;
  }
};

template <typename K, typename V, typename Hash, size_t SIZE>
struct HashMap {
  static constexpr K knull = static_cast<K>(0) - 1;
  static constexpr V vnull = static_cast<V>(0) - 1;

  K *keys_;
  V *vals_;

  __device__
  HashMap(K *keys, V *vals) : keys_(keys), vals_(vals) {}

  __device__
  bool insert(K k, V v) {
    size_t slot = Hash()(k) % SIZE, i = slot;
    do {
      K prev = atomicCAS(&keys_[i], knull, k);
      if (prev == k || prev == knull) {
        vals_[i] = v;
        return true;
      }
      i = (i + 1) % SIZE;
    } while (i != slot);

    return false;
  }

  __device__
  V lookup(K k) {
    size_t slot = Hash()(k) % SIZE, i = slot;
    while (keys_[i] != knull) {
      if (keys_[i] == k) return vals_[i];
      i = (i + 1) % SIZE;
    }
    return vnull;
  }

  __device__
  void blockReset() {
    int tid = threadIdx.x + (threadIdx.y * blockDim.x);
    while (tid < SIZE) {
      keys_[tid] = knull;
      tid += blockDim.x * blockDim.y;
    }
    __syncthreads();
  }
};

template <unsigned STRIDE = 1>
__device__ __forceinline__
void loadBinRange(void *temp, int &beg, int &end, size_t K, size_t numBins, int *ranges) {
  int *tmp = reinterpret_cast<int *>(temp);
  int *begs = tmp;
  int *ends = tmp + blockDim.y;
  if (threadIdx.y == 0 && threadIdx.x < blockDim.y) {
    size_t offsetBin = blockIdx.x * STRIDE;
    size_t offsetBeg = offsetBin * K;
    size_t offsetEnd = min(offsetBin + STRIDE, numBins) * K;
    begs[threadIdx.x] = ranges[offsetBeg + threadIdx.x];
    ends[threadIdx.x] = ranges[offsetEnd + threadIdx.x];
  }
  __syncthreads();

  beg = begs[threadIdx.y];
  end = ends[threadIdx.y];
  __syncthreads();
}

template <typename T, unsigned STRIDE = 1, unsigned TABLESIZE = 1 * 1024>
__global__
void radixJoinAggregateKernel(
  T *keysA, T *valsA, T *keysB, T *valsB, int *rangesA, int *rangesB,
  size_t K, size_t numBins, uint64_t *result) {
  
  __shared__ T tabKeys[TABLESIZE];
  __shared__ T tabVals[TABLESIZE];

  // load bins's begin and end for A and B
  int begA, endA;
  loadBinRange<STRIDE>(&tabKeys, begA, endA, K, numBins, rangesA);

  int begB, endB;
  loadBinRange<STRIDE>(&tabKeys, begB, endB, K, numBins, rangesB);

  // reset hashTable
  using HashMap_t = HashMap<uint64_t, uint64_t, ShiftHash<uint64_t>, TABLESIZE>;
  HashMap_t localMap(tabKeys, tabVals);

  localMap.blockReset();

  // build hash table
  begA += threadIdx.x;
  while (begA < endA) {
    T key = keysA[begA];
    T val = valsA[begA];
    localMap.insert(key, val);
    begA += blockDim.x;
  }
  __syncthreads();

  // prob hash table
  uint64_t localR = 0;
  begB += threadIdx.x;
  while (begB < endB) {
    T key = keysB[begB];
    T val = valsB[begB];
    T r = localMap.lookup(key);
    // assert(val == r);
    localR += (r + val);

    begB += blockDim.x;
  }
  __syncthreads();

  uint64_t *blockR = reinterpret_cast<uint64_t *>(&tabKeys[0]);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    *blockR = 0; 
  }
  __syncthreads();

  atomicAdd(blockR, localR);
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicAdd(result, *blockR);
  }

}

template <typename T, unsigned STRIDE = 1, unsigned TABLESIZE = 1 * 1024>
__global__
void radixJoinOutputKernel(
  T *keysA, T *valsA, T *keysB, T *valsB, int *rangesA, int *rangesB,
  size_t K, size_t numBins, uint64_t *result) {

  __shared__ T tabKeys[TABLESIZE];
  __shared__ T tabVals[TABLESIZE];

  // load bins's begin and end for A and B
  int begA, endA;
  loadBinRange<STRIDE>(&tabKeys, begA, endA, K, numBins, rangesA);

  int begB, endB;
  loadBinRange<STRIDE>(&tabKeys, begB, endB, K, numBins, rangesB);

  // reset hashTable
  using HashMap_t = HashMap<uint64_t, uint64_t, ShiftHash<uint64_t>, TABLESIZE>;
  HashMap_t localMap(tabKeys, tabVals);

  localMap.blockReset();

  // build hash table
  begA += threadIdx.x;
  while (begA < endA) {
    T key = keysA[begA];
    T val = valsA[begA];
    localMap.insert(key, val);
    begA += blockDim.x;
  }
  __syncthreads();

  // prob hash table
  uint64_t localR = 0;
  begB += threadIdx.x;
  while (begB < endB) {
    T key = keysB[begB];
    T val = valsB[begB];
    T r = localMap.lookup(key);
    // assert(val == r);
    localR += (r + val);
    keysB[begB] = r; // materialize the result to gpu memory

    begB += blockDim.x;
  }
  __syncthreads();

  uint64_t *blockR = reinterpret_cast<uint64_t *>(&tabKeys[0]);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    *blockR = 0; 
  }
  __syncthreads();

  atomicAdd(blockR, localR);
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicAdd(result, *blockR);
  }
}

template <typename T, unsigned STRIDE = 1, unsigned TABLESIZE = 2 * 1024>
void radixJoinAggregate(T *keysA, T *valsA, T *keysB, T *valsB, int *rangesA, int *rangesB, int K, size_t numBins, 
  uint64_t *result, hipStream_t s) {
  size_t numBlockX = (numBins + STRIDE - 1) / STRIDE;
  dim3 grid(numBlockX, 1);
  dim3 block(512 / K, K);
  gpuio::hip::LanuchKernel(radixJoinAggregateKernel<T, STRIDE, TABLESIZE>, grid, block, 0, s,
    keysA, valsA, keysB, valsB, rangesA, rangesB, K, numBins, result
  );
}

template <typename T, unsigned STRIDE = 1, unsigned TABLESIZE = 2 * 1024>
void radixJoinOutput(T *keysA, T *valsA, T *keysB, T *valsB, int *rangesA, int *rangesB, int K, size_t numBins, 
  uint64_t *result, hipStream_t s) {
  size_t numBlockX = (numBins + STRIDE - 1) / STRIDE;
  dim3 grid(numBlockX, 1);
  dim3 block(512 / K, K);
  gpuio::hip::LanuchKernel(radixJoinOutputKernel<T, STRIDE, TABLESIZE>, grid, block, 0, s,
    keysA, valsA, keysB, valsB, rangesA, rangesB, K, numBins, result
  );
}




};
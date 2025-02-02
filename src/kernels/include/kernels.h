#pragma once

#include <hipWrapper.h>
#include <hip/hip_runtime.h>

namespace gpuio::kernels::hashtable::hash {

template <typename T>
struct Identity{

  __device__ __host__ __forceinline__ 
  T operator()(T k) {
    return k;
  }

};

struct Murmur{
  // 32 bit Murmur3 hash (https://github.com/nosferalatu/SimpleGPUHashTable/blob/master/src/linearprobing.cu)
  __device__ __host__ __forceinline__ 
  uint32_t operator()(uint32_t k)
  {
      k ^= k >> 16;
      k *= 0x85ebca6b;
      k ^= k >> 13;
      k *= 0xc2b2ae35;
      k ^= k >> 16;
      return k;
  }
};

} // namespace gpuio::kernels::hashtable::hash

namespace gpuio::kernels::hashtable {

// only contains a key array
template <typename K, typename Hash>
struct HashSet {
  static constexpr K null = static_cast<K>(0) - 1;

  K *keys;
  size_t size;

  __host__
  HashSet(gpuio::hip::MemoryRef kmem) {
    keys = reinterpret_cast<K *>(kmem.ptr);
    size = kmem.size / sizeof(K);
  }

  __device__
  bool insert(K k) {
    size_t slot = Hash()(k) % size, i = slot;

    do {
      // the atomic primitve on GPUs only support 32/64 bits
      K prev = atomicCAS(&keys[i], null, k);
      if (prev == null || prev == k) return true;

      i = (i + 1) % size;
    } while (i != slot);

    return false;
  }

  __device__
  bool lookup(K k) {
    size_t slot = Hash()(k) % size, i = slot;

    while (keys[i] != null) {
      if (keys[i] == k) return true;
    }

    return false;
  }
};

template <typename K, typename V>
struct KeyValPair {
  using Key_t = K;
  using Val_t = V;
  K key;
  V val;
};

template <typename K, typename V, typename Hash>
struct HashMap {
  static constexpr K knull = static_cast<K>(0) - 1;
  static constexpr V vnull = static_cast<V>(0) - 1;

  struct Pair {
    K key;
    V val;
  };


  Pair *keyVals;
  size_t size;

  __host__
  HashMap(gpuio::hip::MemoryRef kmem) {
    keyVals = reinterpret_cast<Pair *>(kmem.ptr);
    size = kmem.size / sizeof(Pair);
  }

  __device__
  bool insert(K k, V v) {
    size_t slot = Hash()(k) % size, i = slot;
    do {
      K prev = atomicCAS(&keyVals[i].key, knull, k);
      if (prev == k || prev == knull) {
        keyVals[i] = Pair{k, v};
        return true;
      }

      i = (i + 1) % size;
    } while (i != slot);

    return false;
  }

  __device__
  V lookup(K k) {
    size_t slot = Hash()(k) % size, i = slot;

    while (keyVals[i].key != knull) {
      if (keyVals[i].key == k) return keyVals[i].val;
      i = (i + 1) % size;
    }

    return vnull;
  }
};

/*
 * Build Hash Table
 */

template <typename T, size_t F = 1>
struct Fraction {
  __device__ __host__
  bool operator()(T v) {
    return v % F == 0;
  }
};


template <typename K, typename Hash, typename Cond = Fraction<K>>
__global__
void populateHashSet(HashSet<K, Hash> set, gpuio::hip::MemoryRef keys) {
  K *keysPtr = reinterpret_cast<K *>(keys.ptr);
  size_t numKeys = keys.size / sizeof(K);
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < numKeys) {
    auto k = keysPtr[tid];
    if (Cond()(k)) {
      set.insert(k);
    }
  }
}


template <typename K, typename V, typename Hash, typename Cond = Fraction<K>>
__global__
void populateHashMap(HashMap<K, V, Hash> map, gpuio::hip::MemoryRef keys, gpuio::hip::MemoryRef vals) {
  K *keysPtr = reinterpret_cast<K *>(keys.ptr);
  size_t numKeys = keys.size / sizeof(K);
  V *valsPtr = reinterpret_cast<V *>(vals.ptr);
  size_t numVals = vals.size / sizeof(V);
  assert(numKeys == numVals);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int tc = blockDim.x * gridDim.x;
  while (tid < numKeys) {
    auto k = keysPtr[tid];
    auto v = valsPtr[tid];
    if (Cond()(k)) {
      map.insert(k, v);
    }

    tid += tc; 
  }
}

} // namespace gpuio::kernels::hashtable::hash

namespace gpuio::kernels::hashtable::ops {

template <typename K, typename V, typename Hash>
__global__
void probeAndSum(HashMap<K, V, Hash> map, gpuio::hip::MemoryRef keys, gpuio::hip::MemoryRef vals, uint64_t *out) {
  K *keysPtr = reinterpret_cast<K *>(keys.ptr);
  size_t numKeys = keys.size / sizeof(K);
  V *valsPtr = reinterpret_cast<V *>(vals.ptr);
  size_t numVals = vals.size / sizeof(V);
  assert(numKeys == numVals);

  uint64_t sum = 0;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int tc = blockDim.x * gridDim.x;
  while (tid < numKeys) {
    auto k = keysPtr[tid];
    auto v = valsPtr[tid];
    V vb;
    if ((vb = map.lookup(k)) != map.vnull) {
      sum += vb + v;
    }

    tid += tc; 
  }

  atomicAdd(out, sum);
}

template <typename K>
__global__
void loadAndSum(const uint64_t *__restrict__ vals, const size_t n, uint64_t *out, bool screte) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > n) return;
  uint64_t sum = 0;
  sum = vals[tid];
    // sum += k;

  if (screte) {
    *out = sum;
  }
}

} // namespace gpuio::kernels::hashtable::hash

namespace gpuio::kernels {

__global__ void helloWorld() {
  printf("hello world\n");
}





} // namespace gpuio::kernels
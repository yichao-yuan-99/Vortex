#pragma once

#include <hipWrapper.h>

#include <fmt/core.h>
#include <data/double_buffer.h>


namespace gpuio::kernels::sort {

using gpuio::hip::MemoryRef;

template <typename T, unsigned begin_bit, unsigned end_bit>
struct range_less {
  __device__ __host__
  bool operator()(uint64_t a, uint64_t b) {
    uint64_t mask = ((static_cast<T>(1) << end_bit) - 1) ^ ((static_cast<T>(1) << begin_bit) - 1);
    return ((a & mask) ^ (b & mask));
  }
};


// https://en.cppreference.com/w/cpp/algorithm/lower_bound
template<class ForwardIt, class T = typename std::iterator_traits<ForwardIt>::value_type,
         class Compare>
__device__ __host__
ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp) {
  ForwardIt it;
  typename std::iterator_traits<ForwardIt>::difference_type count, step;
  count = std::distance(first, last);

  while (count > 0) {
    it = first;
    step = count / 2;
    std::advance(it, step);

    if (comp(*it, value)) {
      first = ++it;
      count -= step + 1;
    }
    else
      count = step;
  }

  return first;
}

// https://en.cppreference.com/w/cpp/algorithm/upper_bound
template<class ForwardIt, class T = typename std::iterator_traits<ForwardIt>::value_type,
         class Compare>
__device__ __host__
ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp) {
  ForwardIt it;
  typename std::iterator_traits<ForwardIt>::difference_type count, step;
  count = std::distance(first, last);

  while (count > 0) {
    it = first; 
    step = count / 2;
    std::advance(it, step);

    if (!comp(value, *it)) {
      first = ++it;
      count -= step + 1;
    } 
    else
      count = step;
  }

  return first;
}

template <typename T, unsigned BLKSIZE = 512>
__global__
void findBoundaryKernel(T *vals, size_t size, int *out, unsigned begin_bit, unsigned end_bit) {
  T mask = ((static_cast<T>(1) << end_bit) - 1) ^ ((static_cast<T>(1) << begin_bit) - 1);
  int OUTSIZE = (1 << (end_bit - begin_bit)) + 1;

  __shared__ T buf[BLKSIZE + 1];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < size) {
    buf[threadIdx.x + 1] = vals[tid] & mask;
    if (threadIdx.x == 0) {
      if (tid == 0) {
        buf[0] = 0;
      } else {
        buf[0] = vals[tid - 1] & mask;
      }
    }
    __syncthreads();

    if (buf[threadIdx.x] != buf[threadIdx.x + 1]) {
      out[buf[threadIdx.x + 1]] = tid;
    }
  }

  if (tid == 0) {
    out[0] = 0;
    out[OUTSIZE - 1] = size;
  }

}

template <typename T, unsigned BLKSIZE = 512>
void findBoundary(T *vals, size_t size, int *out, unsigned begin_bit, unsigned end_bit, hipStream_t s) {
  size_t nblock = (size + BLKSIZE - 1) / BLKSIZE;
  gpuio::hip::LanuchKernel(findBoundaryKernel<T>, dim3(nblock), dim3(BLKSIZE), 0, s, vals, size, out, begin_bit, end_bit);
}

template <typename T>
__global__
void patchBoundaryKernel(T *b, size_t N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N) return;

  if (b[tid] == -1) {
    int n = tid + 1;
    T r;
    while ((r = b[n]) == -1) n++;
    b[tid] = r;
  }
}

template <unsigned BLKSIZE = 512>
void patchBoundary(int *b, size_t N, hipStream_t s) {
  size_t nblock = (N + BLKSIZE - 1) / BLKSIZE;
  gpuio::hip::LanuchKernel(patchBoundaryKernel<int>, dim3(nblock), dim3(BLKSIZE), 0, s, b, N);
}

namespace rocm {

template <typename T>
void radix_sort_keys(void *temp_ptr, size_t &temp_size, double_buffer &keys, hipStream_t s = 0, bool check = false); 

template <typename T>
void merge(void *temp_ptr, size_t &temp_size, MemoryRef input1, MemoryRef input2, MemoryRef output, hipStream_t s = 0, bool check = false);

template <typename T>
void mergeLevels(void *temp_ptr, size_t &temp_size, double_buffer &inout, std::vector<size_t> div, hipStream_t s = 0, bool check = false);

template <typename T>
void radix_sort_pairs(
  void *temp, size_t &temp_size, double_buffer &keys, double_buffer &vals, size_t size, 
  unsigned begin_bit = 0, unsigned end_bit = 8 * sizeof(T), hipStream_t s = 0, bool check = false
);

} // namespace rocm

} // namesapce gpuio::kernels::sort
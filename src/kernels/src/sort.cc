// The only reason to separate the template header and the source is
// that vscode cannot properly lint the rocprim

#include <sort/sort.h>

#include <rocprim/rocprim.hpp>
#include <fmt/core.h>

namespace gpuio::kernels::sort {


} // namespace gpuio::kernels::sort

namespace gpuio::kernels::sort::rocm {

template <typename T>
void radix_sort_keys(void *temp_ptr, size_t &temp_size, double_buffer &keys, hipStream_t s, bool check) {
  if (check) {
    size_t need;
    radix_sort_keys<T>(nullptr, need, keys);
    if (need > temp_size) {
      throw std::runtime_error("temporary storage size is too small for rocprim radix sort");
    }
  }
  rocprim::double_buffer<T> raw(keys.current(), keys.alternate());
  size_t N = keys.current().size / sizeof(T);
  auto err = rocprim::radix_sort_keys(temp_ptr, temp_size, raw, N, 0, 8 * sizeof(T), s);
  if (err != hipSuccess) {
    throw std::runtime_error("cannot perform rocprim radix sort");
  }
  if (static_cast<void *>(raw.current()) == keys.alternate()) {
    keys.swap();
  }
}

template void radix_sort_keys<uint64_t>(void *temp_ptr, size_t &temp_size, double_buffer &keys, hipStream_t s, bool check);
template void radix_sort_keys<uint32_t>(void *temp_ptr, size_t &temp_size, double_buffer &keys, hipStream_t s, bool check);
template void radix_sort_keys<int64_t>(void *temp_ptr, size_t &temp_size, double_buffer &keys, hipStream_t s, bool check);
template void radix_sort_keys<int32_t>(void *temp_ptr, size_t &temp_size, double_buffer &keys, hipStream_t s, bool check);

template <typename T>
void merge(void *temp_ptr, size_t &temp_size, MemoryRef input1, MemoryRef input2, MemoryRef output, hipStream_t s, bool check) {
  if (check) {
    size_t need;
    merge<T>(nullptr, need, input1, input2, output);
    if (need > temp_size) {
      throw std::runtime_error("temporary storage size is too small for rocprim merge");
    }
  }

  T *in1 = input1;
  T *in2 = input2;
  T *out = output;
  size_t in1size = input1.size / sizeof(T);
  size_t in2size = input2.size / sizeof(T);

  using BinaryFunction = ::rocprim::less<typename std::iterator_traits<T *>::value_type>;
  auto err = rocprim::merge(temp_ptr, temp_size,
    in1, in2, out, in1size, in2size, BinaryFunction(), s
  );
  if (err != hipSuccess) {
    throw std::runtime_error("cannot get rocprim radix sort temporary size");
  }
}

template void merge<uint64_t>(void *temp_ptr, size_t &temp_size, MemoryRef input1, MemoryRef input2, MemoryRef output, hipStream_t s, bool check);
template void merge<uint32_t>(void *temp_ptr, size_t &temp_size, MemoryRef input1, MemoryRef input2, MemoryRef output, hipStream_t s, bool check);
template void merge<int64_t>(void *temp_ptr, size_t &temp_size, MemoryRef input1, MemoryRef input2, MemoryRef output, hipStream_t s, bool check);
template void merge<int32_t>(void *temp_ptr, size_t &temp_size, MemoryRef input1, MemoryRef input2, MemoryRef output, hipStream_t s, bool check);

template <typename T>
void mergeLevels(void *temp_ptr, size_t &temp_size, double_buffer &inout, std::vector<size_t> div, hipStream_t s, bool check) {
  size_t numChunks = div.size() - 1;
  assert((numChunks & (numChunks - 1)) == 0);

  // assume three levels for now
  for (size_t offset = 1; offset < numChunks; offset *= 2) {
    for (size_t i = 0; i < numChunks; i += offset * 2) {
      auto in1 = inout.current().slice(div[i], div[i + offset]);
      auto in2 = inout.current().slice(div[i + offset], div[i + offset * 2]);
      auto out = inout.alternate().slice(div[i], div[i + offset * 2]);
      merge<T>(temp_ptr, temp_size, in1, in2, out, s, check);
    }
    inout.swap();
  }
  inout.swap();
}

template void mergeLevels<uint64_t>(void *temp_ptr, size_t &temp_size, double_buffer &inout, std::vector<size_t> div, hipStream_t s, bool check);
template void mergeLevels<uint32_t>(void *temp_ptr, size_t &temp_size, double_buffer &inout, std::vector<size_t> div, hipStream_t s, bool check);
template void mergeLevels<int64_t>(void *temp_ptr, size_t &temp_size, double_buffer &inout, std::vector<size_t> div, hipStream_t s, bool check);
template void mergeLevels<int32_t>(void *temp_ptr, size_t &temp_size, double_buffer &inout, std::vector<size_t> div, hipStream_t s, bool check);

template <typename T>
void radix_sort_pairs(
  void *temp, size_t &temp_size, double_buffer &keys, double_buffer &vals, size_t size, 
  unsigned begin_bit, unsigned end_bit, hipStream_t s, bool check
) {
  if (check) {
    size_t need;
    radix_sort_pairs<T>(nullptr, need, keys, vals, size, begin_bit, end_bit);
    if (need > temp_size) {
      throw std::runtime_error("temporary storage is too small for rocm radix sort pairs");
    }
  } 

  rocprim::double_buffer<T> raw_keys(keys.current(), keys.alternate());
  rocprim::double_buffer<T> raw_vals(vals.current(), vals.alternate());
  auto err = rocprim::radix_sort_pairs(temp, temp_size, raw_keys, raw_vals, size, begin_bit, end_bit, s);
  if (err != hipSuccess) {
    throw std::runtime_error("cannot lanuch rocprim radix sort pair temp size");
  }

  if (static_cast<void *>(raw_keys.current()) == keys.alternate()) {
    keys.swap();
  }
  if (static_cast<void *>(raw_vals.current()) == vals.alternate()) {
    vals.swap();
  }
}

template void radix_sort_pairs<uint64_t>(void *temp, size_t &temp_size, double_buffer &keys, double_buffer &vals, size_t size, unsigned begin_bit, unsigned end_bit, hipStream_t s, bool);
template void radix_sort_pairs<uint32_t>(void *temp, size_t &temp_size, double_buffer &keys, double_buffer &vals, size_t size, unsigned begin_bit, unsigned end_bit, hipStream_t s, bool);
template void radix_sort_pairs<int64_t>(void *temp, size_t &temp_size, double_buffer &keys, double_buffer &vals, size_t size, unsigned begin_bit, unsigned end_bit, hipStream_t s, bool);
template void radix_sort_pairs<int32_t>(void *temp, size_t &temp_size, double_buffer &keys, double_buffer &vals, size_t size, unsigned begin_bit, unsigned end_bit, hipStream_t s, bool);

} // namesapce gpuio::kernels::sort::rocm
#include <fmt/core.h>
#include <algorithm>
#include <oneapi/tbb/parallel_sort.h>
#include <thread>
#include <cmath>

#include <execution/execution.h>
#include <execution/ops/sort.h>

using gpuio::execution::double_execution_layout;
using gpuio::execution::StaticPipelineExecutor;
using gpuio::execution::ops::sort::MergeOp;
using gpuio::execution::ops::sort::SortOp;

int main() {
  size_t input_size = 8000'000'000;  
  size_t chunk_size = 1000'000'000;
  using dtype = uint64_t;

  gpuio::hip::DeviceGuard on(0);

  gpuio::hip::Stream s;

  size_t granularity = 40'000'000;

  gpuio::sched::dyn::LoadBalancedExchange exchange(granularity);
  // gpuio::sched::naive::NaiveExchange exchange; // baseline solution, only use one PCIe link

  auto [left, total] = gpuio::hip::MemGetInfo();
  fmt::print("left: {}, total: {}\n", left, total);
  gpuio::hip::DeviceMemory big(left);

  gpuio::hip::HostVector<dtype> data(input_size), alt(input_size);
  gpuio::kernels::double_buffer hostDBuf(data, alt);


  // generate input
  {
    auto randBuf = gpuio::hip::MemoryRef{big}.slice(0, sizeof(dtype) * chunk_size);
    gpuio::utils::rand::fill_rand_host(data, randBuf, 12138);
  }

  // execution
  double_execution_layout layout(big, sizeof(dtype) * chunk_size * 2);

  auto r = gpuio::utils::time::timeit([&] {
    StaticPipelineExecutor<SortOp<dtype>> exec(layout, hostDBuf.current(), hostDBuf.current(), chunk_size);
    exec.run(exchange, s);
  });
  fmt::print("sort, {}\n", r);

  for (int level = 0; level < 3; level++) {
    r = gpuio::utils::time::timeit([&] {
      StaticPipelineExecutor<MergeOp<dtype>> exec(layout, hostDBuf.current(), hostDBuf.alternate(), chunk_size, level);
      exec.run(exchange, s);
      hostDBuf.swap();
    });
    fmt::print("merge, {}\n", r);
  }

  // verification
  {
    auto randBuf = gpuio::hip::MemoryRef{big}.slice(0, sizeof(dtype) * chunk_size);
    gpuio::utils::rand::fill_rand_host(data, randBuf, 12138);
  }

  r = gpuio::utils::time::timeit([&] {
    tbb::parallel_sort(data.begin(), data.end());
  });
  fmt::print("cpu, {}\n", r);

  bool passed = true;
  for (size_t i = 0; i < input_size; i++) {
    auto a = alt[i];
    auto b = data[i];
    if (a != b) {
      fmt::print("{}, {} != {}\n",i, a, b);
      passed = false;
      break;
    }
  }
  fmt::print("passed: {}\n", passed);

}
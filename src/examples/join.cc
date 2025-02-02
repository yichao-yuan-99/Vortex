#include <fmt/core.h>
#include <hipWrapper.h>
#include <sort/sort.h>
#include <io-sched.h>

#include <oneapi/tbb/parallel_sort.h>
#include <oneapi/tbb/parallel_for_each.h>
#include <algorithm>
#include <type_traits>

#include <execution/execution.h>
#include <execution/ops/join.h>

int main() {
  using gpuio::hip::slice_n;
  using gpuio::execution::ops::join::RadixParOp;
  using gpuio::execution::ops::join::RadixJoinOpMeta;
  using gpuio::execution::ops::join::RadixJoinAggregateOp;
  using gpuio::execution::ops::join::RadixJoinOutputOp;
  using gpuio::execution::StaticPipelineExecutor;
  using gpuio::execution::double_execution_layout;


  // --------- Experiment settings
  constexpr size_t table_size = 4000'000'000;
  constexpr char keysAFile[] = "../data/uniqueKeys_uint64_4b_12138_12138.bin";
  constexpr char keysBFile[] = "../data/uniqueKeys_uint64_4b_12138_10086.bin";

  constexpr size_t chunk_size = 500'000'000;
  constexpr size_t granularity = 20'000'000;
  constexpr size_t range_size = (1 << 24) + 1;

  using dtype = uint64_t; // for both key and value
  using JoinOp = RadixJoinAggregateOp<dtype>;
  // using JoinOp = RadixJoinOutputOp<dtype>;

  // --------- Setup data
  gpuio::hip::DeviceGuard on(0);
  gpuio::hip::Stream s;
  gpuio::sched::dyn::LoadBalancedExchange exchange(granularity);
  // gpuio::sched::naive::NaiveExchange exchange; // baseline solution, only use one PCIe link

  gpuio::hip::HostVector<uint32_t> permutation;
  gpuio::hip::HostVector<dtype> keysA(table_size), valsA(table_size);
  gpuio::hip::HostVector<dtype> keysB(table_size), valsB(table_size);

  gpuio::hip::HostVector<int> rangesA(range_size * (table_size / chunk_size));
  gpuio::hip::HostVector<int> rangesB(range_size * (table_size / chunk_size));
  // generate input
  {
    gpuio::utils::io::loadBinary(keysA, keysAFile);
    gpuio::utils::io::loadBinary(keysB, keysBFile);
    gpuio::hip::DeviceMemory randBuf(sizeof(dtype) * 1000'000'000);
    // auto randBuf = gpuio::hip::MemoryRef{mem};
    gpuio::utils::rand::fill_rand_host(valsA, randBuf, 666);
    gpuio::utils::rand::fill_rand_host(valsB, randBuf, 777);
  }

  auto [left, total] = gpuio::hip::MemGetInfo();
  fmt::print("left: {}, total: {}\n", left, total);
  gpuio::hip::DeviceMemory big(left);
  gpuio::hip::MemoryRef bigMem = big;


  uint64_t valsASum = 0, valsBSum = 0, sumAll = 0;
  for (auto v: valsA) {
    valsASum += v;
    sumAll += v;
  }
  for (auto v: valsB) {
    valsBSum += v;
    sumAll += v;
  }

  // ------------ Start Experiment

  double_execution_layout layout(big, chunk_size * sizeof(dtype) * 4 + range_size * sizeof(int));
  auto r = gpuio::utils::time::timeit([&] {
    StaticPipelineExecutor<RadixParOp<dtype>> exec(
      layout, keysA, keysA, valsA, valsA, rangesA, chunk_size, 0, 24
    );
    exec.run(exchange, s);
  });
  fmt::print("parA, {}\n", r);

  r = gpuio::utils::time::timeit([&] {
    StaticPipelineExecutor<RadixParOp<dtype>> exec(
      layout, keysB, keysB, valsB, valsB, rangesB, chunk_size, 0, 24
    );
    exec.run(exchange, s);
  });
  fmt::print("parB, {}\n", r);
  bool passed;
  
  RadixJoinOpMeta meta;
  size_t threshold = 400'000'000 + 1'000;

  r = gpuio::utils::time::timeit([&] {
    meta = RadixJoinOpMeta(rangesA, rangesB, range_size, threshold);
  });
  fmt::print("meta, {}\n", r);

  double_execution_layout layoutN(big, meta.maxPairSizeAligned + meta.maxRangeSizeAligned * 2);
  gpuio::hip::MemsetAsync(layoutN.temp.slice(0, 8), 0, s);
  s.synchronize();

  r = gpuio::utils::time::timeit([&] {
    StaticPipelineExecutor<JoinOp> exec(
      layoutN, keysA, valsA, keysB, valsB, rangesA, rangesB, meta
    );
    exec.run(exchange, s);
  });
  fmt::print("join, {}\n", r);

  fmt::print("================================================================================\n");
  gpuio::hip::HostVector<uint64_t> result(1);
  auto result_d = layoutN.temp.slice(0, 8);
  gpuio::hip::MemcpyAsync(result, result_d, s);
  s.synchronize();
  fmt::print("result: {}\n", result[0]);
  fmt::print("vals A sum: {}\n", valsASum);
  fmt::print("sum of vals in B: {}\n", valsBSum);
  fmt::print("sum all: {}\n", sumAll);

  uint64_t bKeysSum = 0, bValsSum = 0;
  for (auto v: keysB) {
    bKeysSum += v;
  }
  for (auto v: valsB) {
    bValsSum += v;
  }
  fmt::print("sum of B keys now: {}\n", bKeysSum);
  fmt::print("sum of B vals now: {}\n", bValsSum);

  fmt::print("accumulation result is the same: {}\n", result[0] == sumAll);

  if constexpr (std::is_same<JoinOp, RadixJoinOutputOp<dtype>>::value) {
    fmt::print("join output is the same: {}\n", bKeysSum == valsASum);
  }

}

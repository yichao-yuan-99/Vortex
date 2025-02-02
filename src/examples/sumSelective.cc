#include <execution/execution.h>
#include <simple/kernels.h>

#include <fmt/core.h>
#include <thread>

using gpuio::hip::MemoryRef;
using gpuio::hip::HostVector;
using gpuio::hip::DeviceMemory;
using gpuio::execution::double_execution_layout;
using gpuio::execution::StaticPipelineExecutor;
using gpuio::hip::Stream;
using gpuio::hip::MemcpyAsync;
using gpuio::hip::MemsetAsync;

template <typename T>
struct SumSelectiveOp {
  size_t bufSize_;
  std::vector<std::vector<MemoryRef>> inputs_;

  void *temp_ptr_;
  size_t temp_size_;

  MemoryRef pred_d_;
  size_t chunk_size_;

  
  
  SumSelectiveOp(void *temp_ptr, size_t temp_size, MemoryRef pred_d, MemoryRef vals, size_t chunk_size) 
    : bufSize_(chunk_size * sizeof(T)), temp_ptr_(temp_ptr), temp_size_(temp_size), 
      pred_d_(pred_d), chunk_size_(chunk_size) {
    auto p = gpuio::execution::partitionMem(vals, bufSize_);
    for (auto &t: p) {
      inputs_.push_back({t});
    }
  }

  int operator()(MemoryRef mem, int, int, hipStream_t s) {
    auto result_d = reinterpret_cast<uint64_t *>(temp_ptr_);
    gpuio::kernels::simple::sumSelective<T, 256, 4>(pred_d_, mem, chunk_size_, result_d, s);
    return 0;
  }

  std::vector<MemoryRef> inBuf(MemoryRef mem, int, int) { return {mem}; }
  std::vector<MemoryRef> outBuf(MemoryRef mem, int, int) { return {}; }
  std::vector<MemoryRef> &in(int it) { return inputs_[it]; }
  std::vector<MemoryRef> out(int it) { return {}; }
  size_t size() { return inputs_.size(); }
};

std::vector<double> t_zero, t_gpuio;

int main() {

  size_t totalSize = 16'000'000'000;
  size_t loadSize = 4'000'000'000;
  size_t chunkSize = 2'000'000'000;
  size_t granularity = 20'000'000;
  int SEL = 32;
  HostVector<unsigned> vals_l(loadSize), vals;
  HostVector<int> pred(chunkSize);

  gpuio::utils::io::loadBinary(vals_l, "../data/rand_uint32_4b.bin");
  vals.insert(vals.end(), vals_l.begin(), vals_l.end());
  vals.insert(vals.end(), vals_l.begin(), vals_l.end());
  vals.insert(vals.end(), vals_l.begin(), vals_l.end());
  vals.insert(vals.end(), vals_l.begin(), vals_l.end());



  std::vector<int> SELS;
  // int SELL = 1;
  // int SELM = 32;
  int SELL = 32;
  int SELM = 64;
  for (int i = SELL; i < SELM; i++) {
    SELS.push_back(i * 2);
  }

  for (int SEL : SELS) {
    for (int i = 0; i < chunkSize; i++) {
      pred[i] = (vals[i] % SEL) == 0;
    }
    uint64_t groundTruth = 0;

    std::vector<std::thread> ts;
    size_t t_csize = 250'000'000;
    int numt = totalSize / t_csize;
    std::vector<uint64_t> p_gt(numt, 0);
    for (size_t j = 0; j < numt; j++) {
      ts.emplace_back([&, j] {
        for (size_t k = j * t_csize; k < (j + 1) * t_csize; k++) {
          p_gt[j] += pred[k % chunkSize] ? vals[k] : 0;
        }
      });
    }
    for (auto &t: ts) {
      t.join();
    }
    for (auto p_s : p_gt) {
      groundTruth += p_s;
    }
    // for (size_t i = 0; i < totalSize; i++) {
    //   groundTruth += pred[i % chunkSize] ? vals[i] : 0;
    // }


    fmt::print("-------- {:2} --------\n", SEL);
    fmt::print("ground truth for {}: {}\n", SEL, groundTruth);


    gpuio::hip::DeviceGuard on(0);

    gpuio::sched::dyn::LoadBalancedExchange exchange(granularity);

    Stream s;
    DeviceMemory big(2 * chunkSize * sizeof(unsigned) + 100);
    DeviceMemory pred_d(chunkSize * sizeof(int));
    double_execution_layout layout(big, chunkSize * sizeof(int));
    uint64_t *result_d = layout.temp;
    MemsetAsync(layout.temp, 0, s);
    MemcpyAsync(pred_d, pred, s);
    s.synchronize();

    double time;

    // zero-copy
    time = gpuio::utils::time::timeit([&] {
      for (size_t i = 0; i < totalSize; i += chunkSize) {
        unsigned *vals_p = vals.data() + i;
        int *pred_p = pred_d;
        gpuio::kernels::simple::sumSelective<unsigned, 256, 4>(pred_p, vals_p, chunkSize, result_d, s);
      }
      s.synchronize();
    });

    t_zero.push_back(time);

    HostVector<uint64_t> result_h(1);
    MemcpyAsync(result_h, layout.temp.slice(0, sizeof(uint64_t)), s);
    s.synchronize();
    assert(result_h[0] == groundTruth);
    fmt::print("zero copy result: {}\n", result_h[0]);

    // gpu-io
    MemsetAsync(layout.temp, 0, s);
    MemcpyAsync(pred_d, pred, s);
    s.synchronize();

    time = gpuio::utils::time::timeit([&] {
      StaticPipelineExecutor<SumSelectiveOp<unsigned>> exec(layout, pred_d, vals, chunkSize);
      exec.run(exchange, s);
    });

    t_gpuio.push_back(time);

    MemcpyAsync(result_h, layout.temp.slice(0, sizeof(uint64_t)), s);
    s.synchronize();
    assert(result_h[0] == groundTruth);
    fmt::print("gpu io result: {}\n", result_h[0]);

  }

  fmt::print("================================================================================\n");
  
  for (int i = SELL; i < SELM; i++) {
    fmt::print("{}, {}, {}\n", i * 2, t_zero[i - SELL], t_gpuio[i - SELL]);
  }

}
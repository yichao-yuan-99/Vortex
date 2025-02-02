#include <ssb-utils/ssb-utils.h>
#include <fmt/core.h>
#include <fmt/color.h>
#include <execution/execution.h>
#include <crystal/ssb/q1x.h>

using gpuio::hip::MemoryRef;
using gpuio::execution::LinearMemrefAllocator;

struct CrystalQ11Op {
  size_t chunk_cnt_;
  std::vector<std::vector<MemoryRef>> inputs_;
  std::vector<size_t> inputsCnts_;
  ssb::DeviceBuffers &buf_;

  struct Layout {
    MemoryRef lo_orderdate, lo_quantity, lo_extendedprice, lo_discount;
    Layout(MemoryRef mem, size_t chunk_cnt) {
      LinearMemrefAllocator A(mem);
      lo_orderdate = A.alloc(chunk_cnt * sizeof(int));
      lo_quantity = A.alloc(chunk_cnt * sizeof(int));
      lo_extendedprice = A.alloc(chunk_cnt * sizeof(int));
      lo_discount = A.alloc(chunk_cnt * sizeof(int));
    }
  };

  CrystalQ11Op(void *, size_t, ssb::DeviceBuffers &buf, size_t chunk_cnt, ssb::Dataset &data)
    : buf_(buf), chunk_cnt_(chunk_cnt) {
    inputs_ = ssb::loadFactSegments(data, chunk_cnt);
    inputsCnts_ = ssb::loadFactSegmentsCnt(chunk_cnt);
  }

  int operator()(MemoryRef mem, int, int it, hipStream_t s) {
    Layout layout(mem, chunk_cnt_);
    unsigned long long *result = buf_.result;
    crystal::q1x::Q11Run<256, 4>(layout.lo_orderdate, layout.lo_discount, layout.lo_quantity, layout.lo_extendedprice,
      inputsCnts_[it], result, s
    );
    return 0;
  }

  std::vector<MemoryRef> inBuf(MemoryRef mem, int, int) { return std::vector<MemoryRef>{mem}; }
  std::vector<MemoryRef> outBuf(MemoryRef, int, int) { return std::vector<MemoryRef>{}; }
  std::vector<MemoryRef> &in(int it) { return inputs_[it]; }
  std::vector<MemoryRef> out(int) { return std::vector<MemoryRef>{}; }
  size_t size() { return inputs_.size(); }
};

int main() {
  fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::green), "SSB QEURY 1.1\n");
  ssb::Dataset data("/home1/yichaoy/work/dbGen/crystal/test/ssb/data/s1000_columnar", ssb::Q1xDataConfig);
  ssb::printDatasetSummary(data);

  gpuio::hip::DeviceGuard on(0);
  ssb::DeviceBuffers bufs(0);
  gpuio::hip::Stream s;
  size_t granularity = 40'000'000;
  gpuio::sched::dyn::LoadBalancedExchange exchange(granularity);
  // gpuio::sched::naive::NaiveExchange exchange; // baseline solution, only use one PCIe link
  auto [left, total] = gpuio::hip::MemGetInfo();
  fmt::print("[all allocation done] left: {}, total: {}\n", left, total);
  fmt::print("============================================================================================\n");

  { // load dim tables
    double t;
    t = gpuio::utils::time::timeit([&] { 
        auto [dst, src] = ssb::loadDimPlan(bufs, data);
        exchange.launch(dst, src, {}, {});
        exchange.sync();
    }); 

    fmt::print("dim_load_time, {}\n", t);
  }

  size_t chunk_cnt = 400'000'000;
  size_t numCols = 4;

  { // perform computation
    MemoryRef result = MemoryRef{bufs.result}.slice(0, 8);
    double t;
    t = gpuio::utils::time::timeit([&] {
      gpuio::hip::MemsetAsync(result, 0, s);
      s.synchronize();
      gpuio::execution::double_execution_layout layout(bufs.main, chunk_cnt * numCols * sizeof(int));
      gpuio::execution::StaticPipelineExecutor<CrystalQ11Op> exec(layout, bufs, chunk_cnt, data);
      exec.run(exchange, s);
    });
    fmt::print("exec, {}\n", t);

    fmt::print("====================================== RESULT ==============================================\n");
    gpuio::hip::HostVector<uint64_t> r_h(1);
    gpuio::hip::MemcpyAsync(r_h, result, s);
    s.synchronize();
    fmt::print("{}\n", r_h[0]);

  }

}
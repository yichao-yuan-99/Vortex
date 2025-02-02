#include <ssb-utils/ssb-utils.h>
#include <fmt/core.h>
#include <fmt/color.h>
#include <execution/execution.h>
#include <crystal/ssb/q2x.h>

using gpuio::hip::MemoryRef;
using gpuio::execution::LinearMemrefAllocator;

struct CrystalQ22Op {
  size_t chunk_cnt_;
  std::vector<std::vector<MemoryRef>> inputs_;
  std::vector<size_t> inputsCnts_;
  ssb::DeviceBuffers &buf_;

  struct Layout {
    MemoryRef lo_partkey, lo_suppkey, lo_orderdate, lo_revenue;
    Layout(MemoryRef mem, size_t chunk_cnt) {
      LinearMemrefAllocator A(mem);
      lo_partkey = A.alloc(chunk_cnt * sizeof(int));
      lo_suppkey = A.alloc(chunk_cnt * sizeof(int));
      lo_orderdate = A.alloc(chunk_cnt * sizeof(int));
      lo_revenue = A.alloc(chunk_cnt * sizeof(int));
    }
  };

  CrystalQ22Op(void *, size_t, ssb::DeviceBuffers &buf, size_t chunk_cnt, ssb::Dataset &data)
    : buf_(buf), chunk_cnt_(chunk_cnt) {
    inputs_ = ssb::loadFactSegments(data, chunk_cnt);
    inputsCnts_ = ssb::loadFactSegmentsCnt(chunk_cnt);
  }

  int operator()(MemoryRef mem, int, int it, hipStream_t s) {
    Layout layout(mem, chunk_cnt_);
    int *result = buf_.result;
    crystal::q2x::probleTables<256, 4>(
      layout.lo_orderdate, layout.lo_partkey, layout.lo_suppkey, layout.lo_revenue,
      inputsCnts_[it],
      buf_.ht_s, buf_.ht_s.size() / sizeof(int) / 2,
      buf_.ht_p, buf_.ht_p.size() / sizeof(int) / 2,
      buf_.ht_d, buf_.ht_d.size() / sizeof(int) / 2,
      result, s
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
  fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::green), "SSB QEURY 2.2\n");
  ssb::Dataset data("/home1/yichaoy/work/dbGen/crystal/test/ssb/data/s1000_columnar", ssb::Q22DataConfig);
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

  size_t resultCnt = ((1998-1992+1) * (5 * 5 * 40));
  size_t resultSize = resultCnt * 4 * sizeof(int);

  { // perform computation
    MemoryRef result = MemoryRef{bufs.result}.slice(0, resultSize);
    double t;
    t = gpuio::utils::time::timeit([&] {
      gpuio::hip::MemsetAsync(result, 0, s);
      gpuio::hip::MemsetAsync(bufs.ht_s, 0, s);
      gpuio::hip::MemsetAsync(bufs.ht_p, 0, s);
      gpuio::hip::MemsetAsync(bufs.ht_d, 0, s);
      s.synchronize();

      crystal::q2x::buildTables<256, 4, 2>(
        bufs.p_partkey, bufs.p_brand1, bufs.p_category, ssb::p_total_num,
        bufs.d_datekey, bufs.d_year, ssb::d_total_num,
        bufs.s_suppkey, bufs.s_region, ssb::s_total_num, 
        bufs.ht_d, bufs.ht_p, bufs.ht_s, 
        s
      );
      s.synchronize();


      gpuio::execution::double_execution_layout layout(bufs.main, chunk_cnt * numCols * sizeof(int));
      gpuio::execution::StaticPipelineExecutor<CrystalQ22Op> exec(layout, bufs, chunk_cnt, data);
      exec.run(exchange, s);
    });
    fmt::print("exec, {}\n", t);

    fmt::print("====================================== RESULT ==============================================\n");
    gpuio::hip::HostVector<int> r_h(resultCnt * 4);
    gpuio::hip::MemcpyAsync(r_h, result, s);
    s.synchronize();

    
    int res_count = 0;
    for (int i=0; i < resultCnt; i++) {
      unsigned long long t = reinterpret_cast<unsigned long long*>(&r_h[4*i + 2])[0];
      if (r_h[4 * i]) {
        res_count++;
        fmt::print("{:20} {:20} {:20}\n", r_h[4 * i], r_h[4 * i + 1], t);
      }
    }
    fmt::print("--------------------------------------------------------------------------------------------\n");
    fmt::print("{} rows\n", res_count);

  }

}
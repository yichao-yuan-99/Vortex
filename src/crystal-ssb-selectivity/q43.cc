#include <ssb-utils/ssb-utils.h>
#include <fmt/core.h>
#include <fmt/color.h>
#include <execution/execution.h>
#include <crystal-zero-copy/ssb/q43.h>

using gpuio::hip::MemoryRef;
using gpuio::execution::LinearMemrefAllocator;

struct CrystalQ43Op {
  size_t chunk_cnt_;
  std::vector<std::vector<MemoryRef>> inputs_;
  std::vector<size_t> inputsCnts_;
  ssb::DeviceBuffers &buf_;
  ssb::Dataset &data_;

  struct Layout {
    // MemoryRef lo_custkey, lo_partkey, lo_suppkey, lo_orderdate, lo_revenue, lo_supplycost;
    // MemoryRef lo_custkey, lo_partkey, lo_suppkey, lo_orderdate;
    // MemoryRef lo_custkey, lo_partkey, lo_suppkey;
    MemoryRef lo_partkey, lo_suppkey;
    // MemoryRef lo_partkey;
    Layout(MemoryRef mem, size_t chunk_cnt) {
      LinearMemrefAllocator A(mem);
      // lo_custkey = A.alloc(chunk_cnt * sizeof(int));
      lo_partkey = A.alloc(chunk_cnt * sizeof(int));
      lo_suppkey = A.alloc(chunk_cnt * sizeof(int));
      // lo_orderdate = A.alloc(chunk_cnt * sizeof(int));
      // lo_revenue = A.alloc(chunk_cnt * sizeof(int));
      // lo_supplycost = A.alloc(chunk_cnt * sizeof(int));
    }
  };

  CrystalQ43Op(void *, size_t, ssb::DeviceBuffers &buf, size_t chunk_cnt, ssb::Dataset &data)
    : buf_(buf), chunk_cnt_(chunk_cnt), data_(data) {
    std::vector<MemoryRef> cols;
    // cols.push_back(data.lo_custkey);
    cols.push_back(data.lo_partkey);
    cols.push_back(data.lo_suppkey);
    // cols.push_back(data.lo_orderdate);
    inputs_ = ssb::loadFactColsSegments(cols, chunk_cnt);
    inputsCnts_ = ssb::loadFactSegmentsCnt(chunk_cnt);
  }

  int operator()(MemoryRef mem, int, int it, hipStream_t s) {
    Layout layout(mem, chunk_cnt_);
    int *result = buf_.result;

    size_t offset = it * chunk_cnt_;
    int *p_lo_custkey = static_cast<int *>(data_.lo_custkey) + offset;
    // int *p_lo_suppkey = static_cast<int *>(data_.lo_suppkey) + offset;
    int *p_lo_orderdate = static_cast<int *>(data_.lo_orderdate) + offset;
    int *p_lo_revenue = static_cast<int *>(data_.lo_revenue) + offset;
    int *p_lo_supplycost = static_cast<int *>(data_.lo_supplycost) + offset;
    crystal::q43::probleTables<256, 4>(
      p_lo_orderdate, layout.lo_partkey, p_lo_custkey, layout.lo_suppkey, 
      p_lo_revenue, p_lo_supplycost,
      inputsCnts_[it],
      buf_.ht_p, buf_.ht_p.size() / sizeof(int) / 2,
      buf_.ht_s, buf_.ht_s.size() / sizeof(int) / 2,
      buf_.ht_c, buf_.ht_c.size() / sizeof(int) / 2,
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
  fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::green), "SSB QEURY 4.3\n");
  ssb::Dataset data("/home1/yichaoy/work/dbGen/crystal/test/ssb/data/s1000_columnar", ssb::Q43DataConfig);
  ssb::printDatasetSummary(data);

  gpuio::hip::DeviceGuard on(0);
  ssb::DeviceBuffers bufs(0);
  gpuio::hip::Stream s;
  size_t granularity = 40'000'000;
  gpuio::sched::dyn::LoadBalancedExchange exchange(granularity);
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
  size_t numCols = 2;

  size_t resultCnt = ((1998-1992+1) * 250 * 1000);
  size_t resultSize = resultCnt * 6 * sizeof(int);

  { // perform computation
    MemoryRef result = MemoryRef{bufs.result}.slice(0, resultSize);
    double t;
    t = gpuio::utils::time::timeit([&] {
      gpuio::hip::MemsetAsync(result, 0, s);
      gpuio::hip::MemsetAsync(bufs.ht_d, 0, s);
      gpuio::hip::MemsetAsync(bufs.ht_p, 0, s);
      gpuio::hip::MemsetAsync(bufs.ht_s, 0, s);
      gpuio::hip::MemsetAsync(bufs.ht_c, 0, s);
      s.synchronize();

      crystal::q43::buildTables<256, 4>(
        bufs.d_datekey, bufs.d_year, ssb::d_total_num,
        bufs.p_partkey, bufs.p_category, bufs.p_brand1, ssb::p_total_num,
        bufs.s_suppkey, bufs.s_nation, bufs.s_city, ssb::s_total_num, 
        bufs.c_custkey, bufs.c_region, ssb::c_total_num,
        bufs.ht_d, bufs.ht_p, bufs.ht_s, bufs.ht_c, 
        s
      );
      s.synchronize();


      gpuio::execution::double_execution_layout layout(bufs.main, chunk_cnt * numCols * sizeof(int));
      gpuio::execution::StaticPipelineExecutor<CrystalQ43Op> exec(layout, bufs, chunk_cnt, data);
      exec.run(exchange, s);
    });
    fmt::print("exec, {}\n", t);

    fmt::print("====================================== RESULT ==============================================\n");
    gpuio::hip::HostVector<int> r_h(resultCnt * 6);
    gpuio::hip::MemcpyAsync(r_h, result, s);
    s.synchronize();

    
    int res_count = 0;
    for (int i=0; i < resultCnt; i++) {
      unsigned long long t1 = reinterpret_cast<unsigned long long*>(&r_h[6 * i + 4])[0];
      if (r_h[6 * i]) {
        res_count++;
        fmt::print("{:20} {:20} {:20} {:20}\n", r_h[6 * i], r_h[6 * i + 1], r_h[6 * i + 2], t1);
      }
    }
    fmt::print("--------------------------------------------------------------------------------------------\n");
    fmt::print("{} rows\n", res_count);

  }
}
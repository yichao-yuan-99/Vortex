#include <ssb-utils/ssb-utils.h>
#include <fmt/core.h>
#include <fmt/color.h>
#include <execution/execution.h>
#include <crystal-zero-copy/ssb/q41.h>

using gpuio::hip::MemoryRef;
using gpuio::execution::LinearMemrefAllocator;

struct CrystalQ41Op {
  size_t chunk_cnt_;
  std::vector<std::vector<MemoryRef>> inputs_;
  std::vector<size_t> inputsCnts_;
  ssb::DeviceBuffers &buf_;

  struct Layout {
    MemoryRef lo_custkey, lo_partkey, lo_suppkey, lo_orderdate, lo_revenue, lo_supplycost;
    Layout(MemoryRef mem, size_t chunk_cnt) {
      LinearMemrefAllocator A(mem);
      lo_custkey = A.alloc(chunk_cnt * sizeof(int));
      lo_partkey = A.alloc(chunk_cnt * sizeof(int));
      lo_suppkey = A.alloc(chunk_cnt * sizeof(int));
      lo_orderdate = A.alloc(chunk_cnt * sizeof(int));
      lo_revenue = A.alloc(chunk_cnt * sizeof(int));
      lo_supplycost = A.alloc(chunk_cnt * sizeof(int));
    }
  };

  CrystalQ41Op(void *, size_t, ssb::DeviceBuffers &buf, size_t chunk_cnt, ssb::Dataset &data)
    : buf_(buf), chunk_cnt_(chunk_cnt) {
    inputs_ = ssb::loadFactSegments(data, chunk_cnt);
    inputsCnts_ = ssb::loadFactSegmentsCnt(chunk_cnt);
  }

  int operator()(MemoryRef mem, int, int it, hipStream_t s) {
    Layout layout(mem, chunk_cnt_);
    int *result = buf_.result;
    crystal::q41::probleTables<256, 4>(
      layout.lo_orderdate, layout.lo_partkey, layout.lo_custkey, layout.lo_suppkey, 
      layout.lo_revenue, layout.lo_supplycost,
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
  fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::green), "SSB QEURY 4.1\n");
  ssb::Dataset data("/home1/yichaoy/work/dbGen/crystal/test/ssb/data/s1000_columnar", ssb::Q41DataConfig);
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
  size_t numCols = 6;

  size_t resultCnt = ((1998-1992+1) * 25);
  size_t resultSize = resultCnt * 4 * sizeof(int);

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

      crystal::q41::buildTables<256, 4>(
        bufs.d_datekey, bufs.d_year, ssb::d_total_num,
        bufs.p_partkey, bufs.p_mfgr, ssb::p_total_num,
        bufs.s_suppkey, bufs.s_region, ssb::s_total_num, 
        bufs.c_custkey, bufs.c_region, bufs.c_nation, ssb::c_total_num,
        bufs.ht_d, bufs.ht_p, bufs.ht_s, bufs.ht_c, 
        s
      );
      s.synchronize();


      int *lo_orderdate = data.lo_orderdate;
      int *lo_partkey = data.lo_partkey;
      int *lo_custkey = data.lo_custkey;
      int *lo_suppkey = data.lo_suppkey;
      int *lo_revenue = data.lo_revenue;
      int *lo_supplycost = data.lo_supplycost;
      int *result = bufs.result;
      size_t part_lo_len = 1'000'000'000;
      for (size_t offset = 0; offset < ssb::lo_total_num; offset += part_lo_len) {
        int *p_lo_orderdate = lo_orderdate + offset;
        int *p_lo_partkey = lo_partkey + offset;
        int *p_lo_custkey = lo_custkey + offset;
        int *p_lo_suppkey = lo_suppkey + offset;
        int *p_lo_revenue = lo_revenue + offset;
        int *p_lo_supplycost = lo_supplycost + offset;
        size_t p_lo_len = std::min(part_lo_len, static_cast<size_t>(ssb::lo_total_num) - offset);
        crystal::q41::probleTables<256, 4>(
          p_lo_orderdate, p_lo_partkey, p_lo_custkey, p_lo_suppkey, 
          p_lo_revenue, p_lo_supplycost,
          p_lo_len, 
          bufs.ht_p, bufs.ht_p.size() / sizeof(int) / 2,
          bufs.ht_s, bufs.ht_s.size() / sizeof(int) / 2,
          bufs.ht_c, bufs.ht_c.size() / sizeof(int) / 2,
          bufs.ht_d, bufs.ht_d.size() / sizeof(int) / 2,
          result, s
        );
      }
      s.synchronize();
    });
    fmt::print("exec, {}\n", t);

    fmt::print("====================================== RESULT ==============================================\n");
    gpuio::hip::HostVector<int> r_h(resultCnt * 4);
    gpuio::hip::MemcpyAsync(r_h, result, s);
    s.synchronize();

    
    int res_count = 0;
    for (int i=0; i < resultCnt; i++) {
      unsigned long long t1 = reinterpret_cast<unsigned long long*>(&r_h[4 * i + 2])[0];
      if (r_h[4 * i]) {
        res_count++;
        fmt::print("{:20} {:20} {:20}\n", r_h[4 * i], r_h[4 * i + 1], t1);
      }
    }
    fmt::print("--------------------------------------------------------------------------------------------\n");
    fmt::print("{} rows\n", res_count);

  }

}
#include <ssb-utils/ssb-utils.h>
#include <fmt/core.h>
#include <fmt/color.h>
#include <execution/execution.h>
#include <crystal-zero-copy/ssb/q2x.h>

using gpuio::hip::MemoryRef;
using gpuio::execution::LinearMemrefAllocator;

int main() {
  fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::green), "SSB QEURY 2.3\n");
  ssb::Dataset data("/home1/yichaoy/work/dbGen/crystal/test/ssb/data/s1000_columnar", ssb::Q22DataConfig);
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

      crystal::q2x::buildTables<256, 4, 3>(
        bufs.p_partkey, bufs.p_brand1, bufs.p_category, ssb::p_total_num,
        bufs.d_datekey, bufs.d_year, ssb::d_total_num,
        bufs.s_suppkey, bufs.s_region, ssb::s_total_num, 
        bufs.ht_d, bufs.ht_p, bufs.ht_s, 
        s
      );
      s.synchronize();


      int *lo_orderdate = data.lo_orderdate;
      int *lo_partkey = data.lo_partkey;
      int *lo_suppkey = data.lo_suppkey;
      int *lo_revenue = data.lo_revenue;
      int *result = bufs.result;
      size_t part_lo_len = 1'000'000'000;
      for (size_t offset = 0; offset < ssb::lo_total_num; offset += part_lo_len) {
        int *p_lo_orderdate = lo_orderdate + offset;
        int *p_lo_partkey = lo_partkey + offset;
        int *p_lo_suppkey = lo_suppkey + offset;
        int *p_lo_revenue = lo_revenue + offset;
        size_t p_lo_len = std::min(part_lo_len, static_cast<size_t>(ssb::lo_total_num) - offset);
        crystal::q2x::probleTables<256, 4>(
          p_lo_orderdate, p_lo_partkey, p_lo_suppkey, p_lo_revenue,
          p_lo_len,
          bufs.ht_s, bufs.ht_s.size() / sizeof(int) / 2,
          bufs.ht_p, bufs.ht_p.size() / sizeof(int) / 2,
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
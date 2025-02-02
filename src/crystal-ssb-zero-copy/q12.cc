#include <ssb-utils/ssb-utils.h>
#include <fmt/core.h>
#include <fmt/color.h>
#include <execution/execution.h>
#include <crystal-zero-copy/ssb/q1x.h>

using gpuio::hip::MemoryRef;
using gpuio::execution::LinearMemrefAllocator;

int main() {
  fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::green), "SSB QEURY 1.2\n");
  ssb::Dataset data("/home1/yichaoy/work/dbGen/crystal/test/ssb/data/s1000_columnar", ssb::Q1xDataConfig);
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

  { // perform computation
    MemoryRef result = MemoryRef{bufs.result}.slice(0, 8);
    double t;
    t = gpuio::utils::time::timeit([&] {
      gpuio::hip::MemsetAsync(result, 0, s);
      s.synchronize();
      int *lo_orderdate = data.lo_orderdate;
      int *lo_discount = data.lo_discount;
      int *lo_quantity = data.lo_quantity;
      int *lo_extendedprice = data.lo_extendedprice;
      size_t part_lo_len = 1'000'000'000;
      unsigned long long *result = bufs.result;
      for (size_t offset = 0; offset < ssb::lo_total_num; offset += part_lo_len) {
        int *p_lo_orderdate = lo_orderdate + offset;
        int *p_lo_discount = lo_discount + offset;
        int *p_lo_quantity = lo_quantity + offset;
        int *p_lo_extendedprice = lo_extendedprice + offset;
        size_t p_lo_len = std::min(part_lo_len, static_cast<size_t>(ssb::lo_total_num) - offset);
        crystal::q1x::Q12Run<256, 4>(p_lo_orderdate, p_lo_discount, p_lo_quantity, p_lo_extendedprice,
          p_lo_len, result, s
        );
      }
      s.synchronize();
    });
    fmt::print("exec, {}\n", t);

    fmt::print("====================================== RESULT ==============================================\n");
    gpuio::hip::HostVector<uint64_t> r_h(1);
    gpuio::hip::MemcpyAsync(r_h, result, s);
    s.synchronize();
    fmt::print("{}\n", r_h[0]);

  }
}
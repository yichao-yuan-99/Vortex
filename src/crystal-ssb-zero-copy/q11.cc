#include <ssb-utils/ssb-utils.h>
#include <fmt/core.h>
#include <fmt/color.h>
#include <execution/execution.h>
#include <crystal-zero-copy/ssb/q1x.h>

using gpuio::hip::MemoryRef;
using gpuio::execution::LinearMemrefAllocator;

int main() {
  fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::green), "SSB QEURY 1.1\n");
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
        crystal::q1x::Q11Run<256, 4>(p_lo_orderdate, p_lo_discount, p_lo_quantity, p_lo_extendedprice,
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

  // 
  size_t l1 = 0, step = 16;
  size_t l2 = 0;
  int *lo_orderdate = data.lo_orderdate;
  int *lo_quantity = data.lo_quantity;
  int *lo_discount = data.lo_discount;
  for (size_t offset = 0; offset < ssb::lo_total_num; offset += step) {
    size_t cur = std::min(step, (size_t) ssb::lo_total_num - offset);
    bool s = false;
    for (size_t i = 0; i < cur; i++) {
      auto d = lo_orderdate[offset + i];
      if (d > 19930000 && d < 19940000) {
        s = true;
        break;
      }
    }
    if (s) {
      l1++;
    }
  }
  fmt::print("l1 passed: {}\n", l1);

  for (size_t offset = 0; offset < ssb::lo_total_num; offset += step) {
    size_t cur = std::min(step, (size_t) ssb::lo_total_num - offset);
    bool s = false;
    for (size_t i = 0; i < cur; i++) {
      auto d = lo_orderdate[offset + i];
      auto q = lo_quantity[offset + i];
      if ((d > 19930000 && d < 19940000) && q < 25) {
        s = true;
        break;
      }
    }
    if (s) {
      l2++;
    }
  }
  fmt::print("l2 passed: {}\n", l2);

  size_t l3 = 0;
  for (size_t offset = 0; offset < ssb::lo_total_num; offset += step) {
    size_t cur = std::min(step, (size_t) ssb::lo_total_num - offset);
    bool s = false;
    for (size_t i = 0; i < cur; i++) {
      auto d = lo_orderdate[offset + i];
      auto q = lo_quantity[offset + i];
      auto dis = lo_discount[offset + i];
      if ((d > 19930000 && d < 19940000) && q < 25 && (dis >= 1 && dis <=3)) {
        s = true;
        break;
      }
    }
    if (s) {
      l3++;
    }
  }
  fmt::print("l3 passed: {}\n", l3);
}
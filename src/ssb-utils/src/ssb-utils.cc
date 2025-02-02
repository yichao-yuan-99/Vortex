#include <ssb-utils/ssb-utils.h>
#include <execution/execution.h>
#include <fmt/core.h>

namespace {
  template <typename T>
  void printVecVals(T *vals, size_t n) {
    assert(n > 10);
    for (size_t i = 0; i < 3; i++) {
      fmt::print("{:10} ", vals[i]);
    }
    fmt::print("... ");
    for (size_t i = 3; i > 0; i--) {
      fmt::print("{:10} ", vals[n - i]);
    }
    fmt::print("\n");
  }
}

namespace ssb {

Dataset::Dataset(std::filesystem::path root, SSBDataSubsetConfig config) : buf_(160'000'000'000) {
  auto c_custkey_p = root / c_custkey_f;
  auto c_city_p    = root / c_city_f; 
  auto c_nation_p  = root / c_nation_f; 
  auto c_region_p  = root / c_region_f; 

  auto s_suppkey_p = root / s_suppkey_f;
  auto s_city_p    = root / s_city_f;
  auto s_nation_p  = root / s_nation_f;
  auto s_region_p  = root / s_region_f;

  auto p_partkey_p  = root / p_partkey_f;
  auto p_mfgr_p     = root / p_mfgr_f;
  auto p_category_p = root / p_category_f;
  auto p_brand1_p   = root / p_brand1_f;

  auto d_datekey_p          = root / d_datekey_f;
  auto d_year_p             = root / d_year_f;
  auto d_yearmonthnum_p     = root / d_yearmonthnum_f;

  auto lo_custkey_p       = root / lo_custkey_f;
  auto lo_partkey_p       = root / lo_partkey_f;
  auto lo_suppkey_p       = root / lo_suppkey_f;
  auto lo_orderdate_p     = root / lo_orderdate_f;
  auto lo_quantity_p      = root / lo_quantity_f;
  auto lo_extendedprice_p = root / lo_extendedprice_f;
  auto lo_discount_p      = root / lo_discount_f;
  auto lo_revenue_p       = root / lo_revenue_f;
  auto lo_supplycost_p    = root / lo_supplycost_f;

  gpuio::execution::LinearMemrefAllocator allocator(buf_);

  if (config.c_custkey) {
    c_custkey = allocator.alloc(c_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(c_custkey, c_custkey_p);
  }
  if (config.c_city) {
    c_city = allocator.alloc(c_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(c_city, c_city_p);
  }
  if (config.c_nation) {
    c_nation = allocator.alloc(c_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(c_nation, c_nation_p);
  }
  if (config.c_region) {
    c_region = allocator.alloc(c_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(c_region, c_region_p);
  }

  if (config.s_suppkey) {
    s_suppkey = allocator.alloc(s_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(s_suppkey, s_suppkey_p);
  }
  if (config.s_city) {
    s_city = allocator.alloc(s_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(s_city, s_city_p);
  }
  if (config.s_nation) {
    s_nation = allocator.alloc(s_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(s_nation, s_nation_p);
  }
  if (config.s_region) {
    s_region = allocator.alloc(s_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(s_region, s_region_p);
  }

  if (config.p_partkey) {
    p_partkey = allocator.alloc(p_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(p_partkey, p_partkey_p);
  }
  if (config.p_mfgr) {
    p_mfgr = allocator.alloc(p_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(p_mfgr, p_mfgr_p);
  }
  if (config.p_category) {
    p_category = allocator.alloc(p_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(p_category, p_category_p);
  }
  if (config.p_brand1) {
    p_brand1 = allocator.alloc(p_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(p_brand1, p_brand1_p);
  }

  if (config.d_datekey) {
    d_datekey = allocator.alloc(d_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(d_datekey, d_datekey_p);
  }
  if (config.d_year) {
    d_year = allocator.alloc(d_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(d_year, d_year_p);
  }
  if (config.d_yearmonthnum) {
    d_yearmonthnum = allocator.alloc(d_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(d_yearmonthnum, d_yearmonthnum_p);
  }

  if (config.lo_custkey) {
    lo_custkey = allocator.alloc(lo_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(lo_custkey, lo_custkey_p);
  }
  if (config.lo_partkey) {
    lo_partkey = allocator.alloc(lo_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(lo_partkey, lo_partkey_p);
  }
  if (config.lo_suppkey) {
    lo_suppkey = allocator.alloc(lo_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(lo_suppkey, lo_suppkey_p);
  }
  if (config.lo_orderdate) {
    lo_orderdate = allocator.alloc(lo_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(lo_orderdate, lo_orderdate_p);
  }
  if (config.lo_quantity) {
    lo_quantity = allocator.alloc(lo_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(lo_quantity, lo_quantity_p);
  }
  if (config.lo_extendedprice) {
    lo_extendedprice = allocator.alloc(lo_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(lo_extendedprice, lo_extendedprice_p);
  }
  if (config.lo_discount) {
    lo_discount = allocator.alloc(lo_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(lo_discount, lo_discount_p);
  }
  if (config.lo_revenue) {
    lo_revenue = allocator.alloc(lo_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(lo_revenue, lo_revenue_p);
  }
  if (config.lo_supplycost) {
    lo_supplycost = allocator.alloc(lo_total_num * sizeof(int));
    gpuio::utils::io::loadBinary(lo_supplycost, lo_supplycost_p);
  }
}

void printDatasetSummary(Dataset &dataset) {
  fmt::print("====================================== DATA SUMMARY ========================================\n");
  if (dataset.c_custkey.size) {
    assert(dataset.c_custkey.size == (c_total_num * sizeof(int)));
    int *vals = dataset.c_custkey;
    fmt::print("[{:20}] ", "c_custkey");
    printVecVals(vals, dataset.c_custkey.size / sizeof(int));
  }
  if (dataset.c_city.size) {
    assert(dataset.c_city.size == (c_total_num * sizeof(int)));
    int *vals = dataset.c_city;
    fmt::print("[{:20}] ", "c_city");
    printVecVals(vals, dataset.c_city.size / sizeof(int));
  }
  if (dataset.c_nation.size) {
    assert(dataset.c_nation.size == (c_total_num * sizeof(int)));
    int *vals = dataset.c_nation;
    fmt::print("[{:20}] ", "c_nation");
    printVecVals(vals, dataset.c_nation.size / sizeof(int));
  }
  if (dataset.c_region.size) {
    assert(dataset.c_region.size == (c_total_num * sizeof(int)));
    int *vals = dataset.c_region;
    fmt::print("[{:20}] ", "c_region");
    printVecVals(vals, dataset.c_region.size / sizeof(int));
  }

  if (dataset.s_suppkey.size) {
    assert(dataset.s_suppkey.size == (s_total_num * sizeof(int)));
    int *vals = dataset.s_suppkey;
    fmt::print("[{:20}] ", "s_suppkey");
    printVecVals(vals, dataset.s_suppkey.size / sizeof(int));
  }
  if (dataset.s_city.size) {
    assert(dataset.s_city.size == (s_total_num * sizeof(int)));
    int *vals = dataset.s_city;
    fmt::print("[{:20}] ", "s_city");
    printVecVals(vals, dataset.s_city.size / sizeof(int));
  }
  if (dataset.s_nation.size) {
    assert(dataset.s_nation.size == (s_total_num * sizeof(int)));
    int *vals = dataset.s_nation;
    fmt::print("[{:20}] ", "s_nation");
    printVecVals(vals, dataset.s_nation.size / sizeof(int));
  }
  if (dataset.s_region.size) {
    assert(dataset.s_region.size == (s_total_num * sizeof(int)));
    int *vals = dataset.s_region;
    fmt::print("[{:20}] ", "s_region");
    printVecVals(vals, dataset.s_region.size / sizeof(int));
  }

  if (dataset.p_partkey.size) {
    assert(dataset.p_partkey.size == (p_total_num * sizeof(int)));
    int *vals = dataset.p_partkey;
    fmt::print("[{:20}] ", "p_partkey");
    printVecVals(vals, dataset.p_partkey.size / sizeof(int));
  }
  if (dataset.p_mfgr.size) {
    assert(dataset.p_mfgr.size == (p_total_num * sizeof(int)));
    int *vals = dataset.p_mfgr;
    fmt::print("[{:20}] ", "p_mfgr");
    printVecVals(vals, dataset.p_mfgr.size / sizeof(int));
  }
  if (dataset.p_category.size) {
    assert(dataset.p_category.size == (p_total_num * sizeof(int)));
    int *vals = dataset.p_category;
    fmt::print("[{:20}] ", "p_category");
    printVecVals(vals, dataset.p_category.size / sizeof(int));
  }
  if (dataset.p_brand1.size) {
    assert(dataset.p_brand1.size == (p_total_num * sizeof(int)));
    int *vals = dataset.p_brand1;
    fmt::print("[{:20}] ", "p_brand1");
    printVecVals(vals, dataset.p_brand1.size / sizeof(int));
  }

  if (dataset.d_datekey.size) {
    assert(dataset.d_datekey.size == (d_total_num * sizeof(int)));
    int *vals = dataset.d_datekey;
    fmt::print("[{:20}] ", "d_datekey");
    printVecVals(vals, dataset.d_datekey.size / sizeof(int));
  }
  if (dataset.d_year.size) {
    assert(dataset.d_year.size == (d_total_num * sizeof(int)));
    int *vals = dataset.d_year;
    fmt::print("[{:20}] ", "d_year");
    printVecVals(vals, dataset.d_year.size / sizeof(int));
  }
  if (dataset.d_yearmonthnum.size) {
    assert(dataset.d_yearmonthnum.size == (d_total_num * sizeof(int)));
    int *vals = dataset.d_yearmonthnum;
    fmt::print("[{:20}] ", "d_yearmonthnum");
    printVecVals(vals, dataset.d_yearmonthnum.size / sizeof(int));
  }

  if (dataset.lo_custkey.size) {
    assert(dataset.lo_custkey.size == (lo_total_num * sizeof(int)));
    int *vals = dataset.lo_custkey;
    fmt::print("[{:20}] ", "lo_custkey");
    printVecVals(vals, dataset.lo_custkey.size / sizeof(int));
  }
  if (dataset.lo_partkey.size) {
    assert(dataset.lo_partkey.size == (lo_total_num * sizeof(int)));
    int *vals = dataset.lo_partkey;
    fmt::print("[{:20}] ", "lo_partkey");
    printVecVals(vals, dataset.lo_partkey.size / sizeof(int));
  }
  if (dataset.lo_suppkey.size) {
    assert(dataset.lo_suppkey.size == (lo_total_num * sizeof(int)));
    int *vals = dataset.lo_suppkey;
    fmt::print("[{:20}] ", "lo_suppkey");
    printVecVals(vals, dataset.lo_suppkey.size / sizeof(int));
  }
  if (dataset.lo_orderdate.size) {
    assert(dataset.lo_orderdate.size == (lo_total_num * sizeof(int)));
    int *vals = dataset.lo_orderdate;
    fmt::print("[{:20}] ", "lo_orderdate");
    printVecVals(vals, dataset.lo_orderdate.size / sizeof(int));
  }
  if (dataset.lo_quantity.size) {
    assert(dataset.lo_quantity.size == (lo_total_num * sizeof(int)));
    int *vals = dataset.lo_quantity;
    fmt::print("[{:20}] ", "lo_quantity");
    printVecVals(vals, dataset.lo_quantity.size / sizeof(int));
  }
  if (dataset.lo_extendedprice.size) {
    assert(dataset.lo_extendedprice.size == (lo_total_num * sizeof(int)));
    int *vals = dataset.lo_extendedprice;
    fmt::print("[{:20}] ", "lo_extendedprice");
    printVecVals(vals, dataset.lo_extendedprice.size / sizeof(int));
  }
  if (dataset.lo_discount.size) {
    assert(dataset.lo_discount.size == (lo_total_num * sizeof(int)));
    int *vals = dataset.lo_discount;
    fmt::print("[{:20}] ", "lo_discount");
    printVecVals(vals, dataset.lo_discount.size / sizeof(int));
  }
  if (dataset.lo_revenue.size) {
    assert(dataset.lo_revenue.size == (lo_total_num * sizeof(int)));
    int *vals = dataset.lo_revenue;
    fmt::print("[{:20}] ", "lo_revenue");
    printVecVals(vals, dataset.lo_revenue.size / sizeof(int));
  }
  if (dataset.lo_supplycost.size) {
    assert(dataset.lo_supplycost.size == (lo_total_num * sizeof(int)));
    int *vals = dataset.lo_supplycost;
    fmt::print("[{:20}] ", "lo_supplycost");
    printVecVals(vals, dataset.lo_supplycost.size / sizeof(int));
  }
  fmt::print("============================================================================================\n");
}

DeviceBuffers::DeviceBuffers(int device) {
  gpuio::hip::DeviceGuard on(device);

  main = DeviceMemory(30'000'000'000);
  c_custkey = DeviceMemory(c_total_num * sizeof(int));
  c_city = DeviceMemory(c_total_num * sizeof(int));
  c_nation = DeviceMemory(c_total_num * sizeof(int));
  c_region = DeviceMemory(c_total_num * sizeof(int));

  s_suppkey = DeviceMemory(s_total_num * sizeof(int));
  s_city = DeviceMemory(s_total_num * sizeof(int));
  s_nation = DeviceMemory(s_total_num * sizeof(int));
  s_region = DeviceMemory(s_total_num * sizeof(int));

  p_partkey = DeviceMemory(p_total_num * sizeof(int));
  p_mfgr = DeviceMemory(p_total_num * sizeof(int));
  p_category = DeviceMemory(p_total_num * sizeof(int));
  p_brand1 = DeviceMemory(p_total_num * sizeof(int));

  d_datekey = DeviceMemory(d_total_num * sizeof(int));
  d_year = DeviceMemory(d_total_num * sizeof(int));
  d_yearmonthnum = DeviceMemory(d_total_num * sizeof(int));

  ht_c = DeviceMemory(c_total_num * 2 * sizeof(int));
  ht_s = DeviceMemory(s_total_num * 2 * sizeof(int));
  ht_p = DeviceMemory(p_total_num * 2 * sizeof(int));
  ht_d = DeviceMemory((19981230 - 19920101 + 1) * 2 * sizeof(int));

  result = DeviceMemory(100'000'000);
}

auto loadDimPlan(DeviceBuffers &bufs, Dataset &data) -> std::tuple<std::vector<MemoryRef>, std::vector<MemoryRef>> {
  std::vector<MemoryRef> dst, src;
  if (data.c_custkey.size) {
    dst.push_back(bufs.c_custkey);
    src.push_back(data.c_custkey);
  }
  if (data.c_city.size) {
    dst.push_back(bufs.c_city);
    src.push_back(data.c_city);
  }
  if (data.c_nation.size) {
    dst.push_back(bufs.c_nation);
    src.push_back(data.c_nation);
  }
  if (data.c_region.size) {
    dst.push_back(bufs.c_region);
    src.push_back(data.c_region);
  }

  if (data.s_suppkey.size) {
    dst.push_back(bufs.s_suppkey);
    src.push_back(data.s_suppkey);
  }
  if (data.s_city.size) {
    dst.push_back(bufs.s_city);
    src.push_back(data.s_city);
  }
  if (data.s_nation.size) {
    dst.push_back(bufs.s_nation);
    src.push_back(data.s_nation);
  }
  if (data.s_region.size) {
    dst.push_back(bufs.s_region);
    src.push_back(data.s_region);
  }

  if (data.p_partkey.size) {
    dst.push_back(bufs.p_partkey);
    src.push_back(data.p_partkey);
  }
  if (data.p_mfgr.size) {
    dst.push_back(bufs.p_mfgr);
    src.push_back(data.p_mfgr);
  }
  if (data.p_category.size) {
    dst.push_back(bufs.p_category);
    src.push_back(data.p_category);
  }
  if (data.p_brand1.size) {
    dst.push_back(bufs.p_brand1);
    src.push_back(data.p_brand1);
  }

  if (data.d_datekey.size) {
    dst.push_back(bufs.d_datekey);
    src.push_back(data.d_datekey);
  }
  if (data.d_year.size) {
    dst.push_back(bufs.d_year);
    src.push_back(data.d_year);
  }
  if (data.d_yearmonthnum.size) {
    dst.push_back(bufs.d_yearmonthnum);
    src.push_back(data.d_yearmonthnum);
  }
  return {dst, src};
}

auto loadFactColsSegments(std::vector<MemoryRef> &cols, size_t chunk_cnt) -> std::vector<std::vector<MemoryRef>> {
  std::vector<std::vector<MemoryRef>> res;
  assert(cols[0].size == lo_total_num * sizeof(int));
  size_t l = cols[0].size;
  size_t acc = 0, step = chunk_cnt * sizeof(int);
  while (l > 0) {
    size_t t = std::min(step, l);
    std::vector<MemoryRef> r;
    for (auto &c: cols) {
      // [WARNING] should be t, the final step will overflow.
      // but we leave enough space on device and on host, so 
      // ignore it for now.
      r.push_back(c.slice_n(acc, step)); 
    }
    res.push_back(r);

    l -= t;
    acc += t;
  }

  return res;
}

auto loadFactSegments(Dataset &data, size_t chunk_cnt) -> std::vector<std::vector<MemoryRef>> {
  std::vector<MemoryRef> cols;

  if (data.lo_custkey.size) { cols.push_back(data.lo_custkey); }
  if (data.lo_partkey.size) { cols.push_back(data.lo_partkey); }
  if (data.lo_suppkey.size) { cols.push_back(data.lo_suppkey); }
  if (data.lo_orderdate.size) { cols.push_back(data.lo_orderdate); }
  if (data.lo_quantity.size) { cols.push_back(data.lo_quantity); }
  if (data.lo_extendedprice.size) { cols.push_back(data.lo_extendedprice); }
  if (data.lo_discount.size) { cols.push_back(data.lo_discount); }
  if (data.lo_revenue.size) { cols.push_back(data.lo_revenue); }
  if (data.lo_supplycost.size) { cols.push_back(data.lo_supplycost); }

  return loadFactColsSegments(cols, chunk_cnt);
}

auto loadFactSegmentsCnt(size_t chunk_cnt) -> std::vector<size_t> {
  std::vector<size_t> r;
  size_t l = lo_total_num;
  while (l > 0) {
    size_t t = std::min(chunk_cnt, l);
    r.push_back(t);
    l -= t;
  }
  return r;
}

} // namesapce ssb
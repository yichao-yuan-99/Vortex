#pragma once

#include <hipWrapper.h>

#include <filesystem>

#include <ssb-utils/dataset.h>

namespace ssb {

using gpuio::hip::MemoryRef;
using gpuio::hip::DeviceMemory;
using gpuio::hip::HostVector;

struct Dataset {
  MemoryRef c_custkey, c_city, c_nation, c_region;
  MemoryRef s_suppkey, s_city, s_nation, s_region;
  MemoryRef p_partkey, p_mfgr, p_category, p_brand1;
  MemoryRef d_datekey, d_year, d_yearmonthnum;
  MemoryRef lo_custkey, lo_partkey, lo_suppkey, lo_orderdate, lo_quantity; 
  MemoryRef lo_extendedprice, lo_discount, lo_revenue, lo_supplycost;

  HostVector<uint8_t> buf_;

  Dataset(std::filesystem::path root, SSBDataSubsetConfig config);
};

void printDatasetSummary(Dataset &dataset);

struct DeviceBuffers {
  DeviceMemory main;
  DeviceMemory c_custkey, c_city, c_nation, c_region;
  DeviceMemory s_suppkey, s_city, s_nation, s_region;
  DeviceMemory p_partkey, p_mfgr, p_category, p_brand1;
  DeviceMemory d_datekey, d_year, d_yearmonthnum;
  DeviceMemory ht_c, ht_s, ht_p, ht_d;
  DeviceMemory result;

  DeviceBuffers(int device);
};

auto loadDimPlan(DeviceBuffers &bufs, Dataset &data) -> std::tuple<std::vector<MemoryRef>, std::vector<MemoryRef>>;

auto loadFactColsSegments(std::vector<MemoryRef> &cols, size_t chunk_cnt) -> std::vector<std::vector<MemoryRef>>;
auto loadFactSegments(Dataset &data, size_t chunk_cnt) -> std::vector<std::vector<MemoryRef>>;
auto loadFactSegmentsCnt(size_t chunk_cnt) -> std::vector<size_t>;

} // namespace ssb
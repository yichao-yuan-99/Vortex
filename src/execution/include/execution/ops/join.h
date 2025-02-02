#pragma once

#include <join/join.h>

namespace gpuio::execution::ops::join {

template <typename T>
struct RadixParOp {
  size_t bufSize_, chunkSize_, rangeSize_;
  std::vector<std::vector<MemoryRef>> inputs_, outputs_;

  void *temp_ptr_;
  size_t temp_size_;

  unsigned begin_bit_, end_bit_;

  struct Layout {
    double_buffer val, key;
    MemoryRef ranges;
    LinearMemrefAllocator alloc;

    static std::tuple<int, int> dec(int type) { return {(type >> 1) & 1, type & 1}; }
    static int enc(int keyCur, int valCur) { return (keyCur << 1) | valCur; }

    Layout(MemoryRef mem, size_t size, int type) : alloc(mem) {
      auto [keyCur, valCur] = dec(type);

      MemoryRef cur, alt;
      cur = alloc.alloc(size);
      alt = alloc.alloc(size);
      key = double_buffer{cur, alt, keyCur};

      cur = alloc.alloc(size);
      alt = alloc.alloc(size);
      val = double_buffer{cur, alt, valCur};

      assert(alloc.free() == ((1 << 24) + 1) * sizeof(int));
      ranges = alloc.alloc(alloc.free());
    }
    int code() { return enc(key.cur_, val.cur_); }
  };

  static size_t rangeSize(unsigned begin_bit, unsigned end_bit) {
    return ((static_cast<size_t>(1) << (end_bit - begin_bit)) + 1) * sizeof(int);
  }

  RadixParOp(
    void *temp_ptr, size_t temp_size, MemoryRef keyIn, MemoryRef keyOut, 
    MemoryRef valIn, MemoryRef valOut, MemoryRef rangesOut, size_t chunk_size,
    unsigned begin_bit, unsigned end_bit
  ) : bufSize_(chunk_size * sizeof(T)), chunkSize_(chunk_size), 
    rangeSize_(rangeSize(begin_bit, end_bit)),
    temp_ptr_(temp_ptr), temp_size_(temp_size), 
    begin_bit_(begin_bit), end_bit_(end_bit) {
    auto keyInChunks = partitionMem(keyIn, bufSize_);
    auto keyOutChunks = partitionMem(keyOut, bufSize_);
    auto valInChunks = partitionMem(valIn, bufSize_);
    auto valOutChunks = partitionMem(valOut, bufSize_);
    auto rangesChunks = partitionMem(rangesOut, rangeSize_);

    assert(valInChunks.size() == valOutChunks.size() && valOutChunks.size() == keyInChunks.size());
    assert(keyInChunks.size() == keyOutChunks.size() && keyOutChunks.size() == rangesChunks.size());
    for (size_t i = 0; i < keyInChunks.size(); i++) {
      inputs_.emplace_back(std::vector<MemoryRef>{keyInChunks[i], valInChunks[i]});
      outputs_.emplace_back(std::vector<MemoryRef>{keyOutChunks[i], valOutChunks[i], rangesChunks[i]});
    }
  }

  int operator()(MemoryRef mem, int type, int it, hipStream_t s) {
    Layout l(mem, bufSize_, type);
    l.key.swap();
    l.val.swap();
    gpuio::kernels::join::radix_partition<T> (
      temp_ptr_, temp_size_, l.key, l.val, l.ranges, chunkSize_, begin_bit_, end_bit_, s, true
    );
    return l.code();
  }


  std::vector<MemoryRef> inBuf(MemoryRef mem, int type, int) { 
    Layout l(mem, bufSize_, type);
    return {l.key.alternate(), l.val.alternate()};
  }
  std::vector<MemoryRef> outBuf(MemoryRef mem, int type, int) { 
    Layout l(mem, bufSize_, type);
    return {l.key.current(), l.val.current(), l.ranges};
  }

  std::vector<MemoryRef> &in(int it) { return inputs_[it]; }
  std::vector<MemoryRef> &out(int it) { return outputs_[it]; }
  size_t size() { return inputs_.size(); }
};


template <typename T>
constexpr T rangeMask(unsigned begin_bit, unsigned end_bit) {
  return ((static_cast<T>(1) << end_bit) - 1) ^ ((static_cast<T>(1) << begin_bit) - 1); 
}

constexpr size_t rangeSize(unsigned begin_bit, unsigned end_bit) {
  return static_cast<size_t>(1) << (end_bit - begin_bit);
}

template <typename T>
size_t sizeOfPartitions(const T &rangesA, const T &rangesB, size_t rangeSize, size_t pos) {
  size_t N = rangesA.size() / rangeSize;
  size_t r = 0;
  for (size_t i = 0; i < N; i++) {
    size_t offset = i * rangeSize;
    r += rangesA[offset + pos];
    r += rangesB[offset + pos];
  }
  return r;
}

template <typename T>
size_t dividePartitions(const T &rangesA, const T &rangesB, size_t rangeSize, size_t th) {
  // find lower bound of th
  size_t beg = 0, end = rangeSize;
  while (end > beg) {
    size_t mid = (end + beg) / 2;
    if (sizeOfPartitions(rangesA, rangesB, rangeSize, mid) < th) {
      beg = mid + 1;
    } else {
      end = mid;
    }
  }

  return beg;
}

struct ParGroup {
  size_t beg, end, n;
};

template <typename T>
auto groupPartitions(
  const T &rangesA, const T &rangesB, size_t rangeSize, size_t bufN) {
  size_t totalN = 0;
  size_t N = rangesA.size() / rangeSize;
  for (size_t i = 0; i < N; i++) {
    size_t offset = i * rangeSize;
    totalN += rangesA[offset + rangeSize - 1];
    totalN += rangesB[offset + rangeSize - 1];
  }

  std::vector<ParGroup> r;
  size_t accN = 0, lastP = 0;
  while (accN < totalN) {
    size_t curP = dividePartitions(rangesA, rangesB, rangeSize, accN + bufN);
    size_t curN = sizeOfPartitions(rangesA, rangesB, rangeSize, curP - 1) - accN;
    r.emplace_back(ParGroup{lastP, curP, curN});
    accN += curN;
    lastP = curP - 1;
  }

  return r;
}

template <typename T>
auto groupPartitionsNaive(
  const T &rangesA, const T &rangesB, size_t rangeSize, size_t bufN) {
  std::vector<size_t> rangesSum(rangeSize);
  size_t N = rangesA.size() / rangeSize;
  for (size_t i = 0; i < N; i++) {
    size_t offset = i * rangeSize;
    for (size_t j = 0; j < rangeSize; j++) {
      rangesSum[j] += rangesA[offset + j];
      rangesSum[j] += rangesB[offset + j];
    } 
  }
  size_t totalN = rangesSum.back();
  // assert(totalN == rangesSum.back());

  std::vector<ParGroup> r;
  size_t accN = 0, lastP = 0;
  while (accN < totalN) {
    size_t curP = lastP;
    for (; curP < rangeSize; curP++) {
      if (rangesSum[curP] - accN > bufN) break;
    }

    if (curP == rangeSize) {
      r.emplace_back(ParGroup{lastP, rangeSize, totalN - accN});
      accN = totalN;
    } else {
      r.emplace_back(ParGroup{lastP, curP, rangesSum[curP - 1] - accN});
      lastP = curP - 1;
      accN = rangesSum[curP - 1];
    }
  }

  return r;
}


using gpuio::hip::MemoryRef;
using gpuio::hip::HostVector;
using gpuio::hip::slice_n;
using gpuio::hip::slice;

struct RadixJoinOpMeta {
  static constexpr size_t alignBound = 10'000'000;
  size_t dataRangeX = 0;                          // # of element in each range chunk (X axis for range)
  size_t dataRangeXSize = 0;                      // size in byte for each range chunk (size of X)
  size_t dataPairX = 0;                           // the # of element in each pair Chunk [FIXME: assume divisble]
  size_t K = 0;                                   // # of chunks for the data (Y axis)

  size_t opPairThreshold = 0;                     // maximum allowed element in each pair chunk

  std::vector<ParGroup> plan;
  size_t maxRangeCnt = 0;                         // maximum of # of elements in a range chunk
  size_t maxRangeSize = 0;                        // Size of above
  size_t maxRangeSizeAligned = 0;                 // align above to alignBound
  size_t maxPairCnt = 0;                          // maximum of # of elements in a pair chunk (2 tables)
  size_t maxPairSize = 0;                         // Size of above
  size_t maxPairSizeAligned = 0;                  // align above to alignBound
  size_t opChunks = 0;                            // # of chunks for the current op

  RadixJoinOpMeta() = default;

  RadixJoinOpMeta(
    HostVector<int> &rangesA, HostVector<int> &rangesB, size_t rangeSize, size_t threshold
  ) : dataRangeX(rangeSize), dataRangeXSize(rangeSize * sizeof(int)), opPairThreshold(threshold) {
    assert(rangesA.size() == rangesB.size() && rangesA.size() % rangeSize == 0);
    dataPairX = rangesA.back();
    K = rangesA.size() / rangeSize;
    // plan = groupPartitionsNaive(rangesA, rangesB, rangeSize, threshold);
    plan = groupPartitions(rangesA, rangesB, rangeSize, threshold);
    for (auto &g: plan) {
      size_t curRangeCnt = (g.end - g.beg) * K;
      size_t curRangeSize = curRangeCnt * sizeof(int);
      size_t curPairCnt = g.n;
      size_t curPairSize = curPairCnt * sizeof(uint64_t) * 2;

      maxRangeCnt = std::max(maxRangeCnt, curRangeCnt);
      maxRangeSize = std::max(maxRangeSize, curRangeSize);
      maxPairCnt = std::max(maxPairCnt, curPairCnt);
      maxPairSize = std::max(maxPairSize, curPairSize);
    }

    maxPairSizeAligned = (maxPairSize + 10'000'000 - 1) / 10'000'000 * 10'000'000;
    maxRangeSizeAligned = (maxRangeSize + 10'000'000 - 1) / 10'000'000 * 10'000'000;
    opChunks = plan.size();
  }
};

void cpuTransRef(HostVector<int> &in, HostVector<int> &out, RadixJoinOpMeta &meta) {
  auto &plan = meta.plan;
  for (size_t i = 0; i < plan.size(); i++) {
    auto &g = plan[i];
    for (size_t x = g.beg; x < g.end; x++) {
      for (size_t y = 0; y < meta.K; y++) {
        int e = in[y * meta.dataRangeX + x];
        int eRef = in[y * meta.dataRangeX + g.beg];
        size_t idxOut = (x + i) * meta.K + y;
        out[idxOut]  = e - eRef;
      }
    }
  }
}

template <typename T>
auto groupValsByRange(HostVector<T> &val, HostVector<int> &ranges, RadixJoinOpMeta &meta) {
  std::vector<std::vector<MemoryRef>> r;
  auto &plan = meta.plan;
  for (size_t i = 0; i < plan.size(); i++) {
    std::vector<MemoryRef> curR;

    auto &g = plan[i];
    for (size_t y = 0; y < meta.K; y++) {
      size_t roffset = y * meta.dataRangeX;
      size_t voffset = y * meta.dataPairX;
      size_t beg = ranges[roffset + g.beg], end = ranges[roffset + g.end - 1];
      curR.push_back(slice_n(val, voffset + beg, end - beg)) ;
    }

    r.push_back(curR);
  }
  return r;
}

template <typename T>
auto groupRanges(HostVector<T> &ranges, RadixJoinOpMeta &meta) {
  std::vector<std::vector<MemoryRef>> r;
  auto &plan = meta.plan;
  for (size_t i = 0; i < plan.size(); i++) {
    auto &g = plan[i];
    std::vector<MemoryRef> curR;
    for (size_t offset = 0; offset < ranges.size(); offset += meta.dataRangeX) {
      curR.push_back(slice_n(ranges, offset + g.beg, g.end - g.beg));
    }
    r.push_back(curR);
  }
  return r;
}



template <typename T>
struct RadixJoinAggregateOp {
  RadixJoinOpMeta &meta_;
  std::vector<std::vector<MemoryRef>> inputs_, outputs_;
  std::vector<size_t> dataSizes_;

  // Layout for the temporary
  struct TempLayout {
    uint64_t *result;
    void *ranges;
    size_t rangesSize;
    TempLayout(void *temp_ptr, size_t temp_size) {
      uint8_t *bytes = reinterpret_cast<uint8_t *>(temp_ptr);
      result = reinterpret_cast<uint64_t *>(bytes);
      ranges = reinterpret_cast<void *>(bytes + 8);
      rangesSize = temp_size - 8;
    }
  };
  TempLayout tempLayout_;

  // Layout for IO transfer
  // match each host memory ref with a ref on device
  std::vector<MemoryRef> bufLayout(MemoryRef mem, std::vector<MemoryRef> &hostRefs) {
    size_t bound = meta_.maxPairSizeAligned;
    auto memData = mem.slice(0, bound);
    auto memRanges = mem.slice(bound, mem.size);
    std::vector<MemoryRef> r;

    assert(hostRefs.size() == meta_.K * 6);

    size_t dataRefNum = meta_.K * 4;
    {
      gpuio::execution::LinearMemrefAllocator a(memData);
      for (size_t i = 0; i < dataRefNum; i++) {
        r.push_back(a.alloc(hostRefs[i].size));
      }
    }
    {
      gpuio::execution::LinearMemrefAllocator a(memRanges);
      for (size_t i = dataRefNum; i < hostRefs.size(); i++) {
        r.push_back(a.alloc(hostRefs[i].size));
      }
    }
    return r;
  }

  // Parameters for kernel lanuch
  struct LaunchParam {
    size_t numA, numB, numBins;
  };
  std::vector<LaunchParam> launchParams_;

  // Layout for kernel execution
  struct DataLayout {
    MemoryRef keysA, valsA, keysB, valsB, rangesA, rangesB;
    DataLayout(MemoryRef mem, size_t bound, size_t numA, size_t numB, size_t numBins, size_t K) {
      auto memData = mem.slice(0, bound);
      auto memBins = mem.slice(bound, mem.size);
      {
        gpuio::execution::LinearMemrefAllocator a(memData);
        keysA = a.alloc(numA * sizeof(T));
        valsA = a.alloc(numA * sizeof(T));
        keysB = a.alloc(numB * sizeof(T));
        valsB = a.alloc(numB * sizeof(T));
      }
      {
        gpuio::execution::LinearMemrefAllocator a(memBins);
        rangesA = a.alloc((numBins + 1) * K * sizeof(int));
        rangesB = a.alloc((numBins + 1) * K * sizeof(int));
      }
    }
  };

  

  RadixJoinAggregateOp(void *temp_ptr, size_t temp_size,
    HostVector<T> &keysA, HostVector<T> &valsA,
    HostVector<T> &keysB, HostVector<T> &valsB,
    HostVector<int> &rangesA, HostVector<int> &rangesB, RadixJoinOpMeta &meta)
    : meta_(meta), tempLayout_(temp_ptr, temp_size) {

    auto keysAGroup = groupValsByRange(keysA, rangesA, meta);
    auto valsAGroup = groupValsByRange(valsA, rangesA, meta);
    auto keysBGroup = groupValsByRange(keysB, rangesB, meta);
    auto valsBGroup = groupValsByRange(valsB, rangesB, meta);
    auto rangesAGroup = groupRanges(rangesA, meta);
    auto rangesBGroup = groupRanges(rangesB, meta);

    auto refsSum = [](std::vector<MemoryRef>& mems) {
      size_t s = 0;
      for (auto m: mems) {
        s += m.size;
      }
      return s;
    };

    for (size_t i = 0; i < meta.opChunks; i++) {
      auto &gAkeys = keysAGroup[i];
      auto &gBkeys = keysBGroup[i];
      auto &rg = meta_.plan[i];
      launchParams_.push_back(LaunchParam{refsSum(gAkeys) / sizeof(T), refsSum(gBkeys) / sizeof(T), rg.end - rg.beg - 1});
    }
    for (size_t i = 0; i < launchParams_.size(); i++) {
      auto &param = launchParams_[i];
    }

    for (size_t i = 0; i < meta.opChunks; i++) {
      std::vector<MemoryRef> curRef;
      curRef.insert(curRef.end(), keysAGroup[i].begin(), keysAGroup[i].end());
      curRef.insert(curRef.end(), valsAGroup[i].begin(), valsAGroup[i].end());
      curRef.insert(curRef.end(), keysBGroup[i].begin(), keysBGroup[i].end());
      curRef.insert(curRef.end(), valsBGroup[i].begin(), valsBGroup[i].end());
      curRef.insert(curRef.end(), rangesAGroup[i].begin(), rangesAGroup[i].end());
      curRef.insert(curRef.end(), rangesBGroup[i].begin(), rangesBGroup[i].end());
      inputs_.push_back(curRef);
    }
  }

  int operator()(MemoryRef mem, int type, int it, hipStream_t s) {
    auto &param = launchParams_[it];
    DataLayout layout(mem, meta_.maxPairSizeAligned, param.numA, param.numB, param.numBins, meta_.K);
    int *rangesA = layout.rangesA;
    int *rangesB = layout.rangesB;
    gpuio::kernels::join::regulateRanges(tempLayout_.ranges, tempLayout_.rangesSize, rangesA, rangesA, param.numBins + 1, meta_.K, s);
    gpuio::kernels::join::regulateRanges(tempLayout_.ranges, tempLayout_.rangesSize, rangesB, rangesB, param.numBins + 1, meta_.K, s);
    gpuio::kernels::join::radixJoinAggregate<T>(layout.keysA, layout.valsA, layout.keysB, layout.valsB, layout.rangesA, layout.rangesB,
      meta_.K, param.numBins, tempLayout_.result, s);
    return type;
  }

  std::vector<MemoryRef> inBuf(MemoryRef mem, int, int it) {
    return bufLayout(mem, in(it));
  }
  std::vector<MemoryRef> outBuf(MemoryRef mem, int, int it) {
    return std::vector<MemoryRef>{};
  }

  std::vector<MemoryRef> &in(int it) { return inputs_[it]; }
  std::vector<MemoryRef> out(int it) { return std::vector<MemoryRef>{};}
  size_t size() {return inputs_.size(); }
};

template <typename T>
struct RadixJoinOutputOp {
  RadixJoinOpMeta &meta_;
  std::vector<std::vector<MemoryRef>> inputs_, outputs_;
  std::vector<size_t> dataSizes_;

  // Layout for the temporary
  struct TempLayout {
    uint64_t *result;
    void *ranges;
    size_t rangesSize;
    TempLayout(void *temp_ptr, size_t temp_size) {
      uint8_t *bytes = reinterpret_cast<uint8_t *>(temp_ptr);
      result = reinterpret_cast<uint64_t *>(bytes);
      ranges = reinterpret_cast<void *>(bytes + 8);
      rangesSize = temp_size - 8;
    }
  };
  TempLayout tempLayout_;

  // Layout for IO transfer
  // match each host memory ref with a ref on device
  std::vector<MemoryRef> bufLayout(MemoryRef mem, std::vector<MemoryRef> &hostRefs) {
    size_t bound = meta_.maxPairSizeAligned;
    auto memData = mem.slice(0, bound);
    auto memRanges = mem.slice(bound, mem.size);
    std::vector<MemoryRef> r;

    assert(hostRefs.size() == meta_.K * 6);

    size_t dataRefNum = meta_.K * 4;
    {
      gpuio::execution::LinearMemrefAllocator a(memData);
      for (size_t i = 0; i < dataRefNum; i++) {
        r.push_back(a.alloc(hostRefs[i].size));
      }
    }
    {
      gpuio::execution::LinearMemrefAllocator a(memRanges);
      for (size_t i = dataRefNum; i < hostRefs.size(); i++) {
        r.push_back(a.alloc(hostRefs[i].size));
      }
    }
    return r;
  }

  // Parameters for kernel lanuch
  struct LaunchParam {
    size_t numA, numB, numBins;
  };
  std::vector<LaunchParam> launchParams_;

  // Layout for kernel execution
  struct DataLayout {
    MemoryRef keysA, valsA, keysB, valsB, rangesA, rangesB;
    DataLayout(MemoryRef mem, size_t bound, size_t numA, size_t numB, size_t numBins, size_t K) {
      auto memData = mem.slice(0, bound);
      auto memBins = mem.slice(bound, mem.size);
      {
        gpuio::execution::LinearMemrefAllocator a(memData);
        keysA = a.alloc(numA * sizeof(T));
        valsA = a.alloc(numA * sizeof(T));
        keysB = a.alloc(numB * sizeof(T));
        valsB = a.alloc(numB * sizeof(T));
      }
      {
        gpuio::execution::LinearMemrefAllocator a(memBins);
        rangesA = a.alloc((numBins + 1) * K * sizeof(int));
        rangesB = a.alloc((numBins + 1) * K * sizeof(int));
      }
    }
  };

  

  RadixJoinOutputOp(void *temp_ptr, size_t temp_size,
    HostVector<T> &keysA, HostVector<T> &valsA,
    HostVector<T> &keysB, HostVector<T> &valsB,
    HostVector<int> &rangesA, HostVector<int> &rangesB, RadixJoinOpMeta &meta)
    : meta_(meta), tempLayout_(temp_ptr, temp_size) {

    auto keysAGroup = groupValsByRange(keysA, rangesA, meta);
    auto valsAGroup = groupValsByRange(valsA, rangesA, meta);
    auto keysBGroup = groupValsByRange(keysB, rangesB, meta);
    auto valsBGroup = groupValsByRange(valsB, rangesB, meta);
    auto rangesAGroup = groupRanges(rangesA, meta);
    auto rangesBGroup = groupRanges(rangesB, meta);

    auto refsSum = [](std::vector<MemoryRef>& mems) {
      size_t s = 0;
      for (auto m: mems) {
        s += m.size;
      }
      return s;
    };

    for (size_t i = 0; i < meta.opChunks; i++) {
      auto &gAkeys = keysAGroup[i];
      auto &gBkeys = keysBGroup[i];
      auto &rg = meta_.plan[i];
      launchParams_.push_back(LaunchParam{refsSum(gAkeys) / sizeof(T), refsSum(gBkeys) / sizeof(T), rg.end - rg.beg - 1});
    }
    for (size_t i = 0; i < launchParams_.size(); i++) {
      auto &param = launchParams_[i];
    }

    for (size_t i = 0; i < meta.opChunks; i++) {
      std::vector<MemoryRef> curRef, curOutRef;
      curRef.insert(curRef.end(), keysAGroup[i].begin(), keysAGroup[i].end());
      curRef.insert(curRef.end(), valsAGroup[i].begin(), valsAGroup[i].end());
      curRef.insert(curRef.end(), keysBGroup[i].begin(), keysBGroup[i].end());
      curRef.insert(curRef.end(), valsBGroup[i].begin(), valsBGroup[i].end());
      curRef.insert(curRef.end(), rangesAGroup[i].begin(), rangesAGroup[i].end());
      curRef.insert(curRef.end(), rangesBGroup[i].begin(), rangesBGroup[i].end());
      inputs_.push_back(curRef);

      curOutRef.insert(curOutRef.end(), keysBGroup[i].begin(), keysBGroup[i].end());
      curOutRef.insert(curOutRef.end(), valsBGroup[i].begin(), valsBGroup[i].end());
      outputs_.push_back(curOutRef);
    }
  }

  int operator()(MemoryRef mem, int type, int it, hipStream_t s) {
    auto &param = launchParams_[it];
    DataLayout layout(mem, meta_.maxPairSizeAligned, param.numA, param.numB, param.numBins, meta_.K);
    int *rangesA = layout.rangesA;
    int *rangesB = layout.rangesB;
    gpuio::kernels::join::regulateRanges(tempLayout_.ranges, tempLayout_.rangesSize, rangesA, rangesA, param.numBins + 1, meta_.K, s);
    gpuio::kernels::join::regulateRanges(tempLayout_.ranges, tempLayout_.rangesSize, rangesB, rangesB, param.numBins + 1, meta_.K, s);
    gpuio::kernels::join::radixJoinOutput<T>(layout.keysA, layout.valsA, layout.keysB, layout.valsB, layout.rangesA, layout.rangesB,
      meta_.K, param.numBins, tempLayout_.result, s);
    return type;
  }

  std::vector<MemoryRef> inBuf(MemoryRef mem, int, int it) {
    return bufLayout(mem, in(it));
  }
  std::vector<MemoryRef> outBuf(MemoryRef mem, int, int it) {
    auto layout = bufLayout(mem, in(it));
    std::vector<MemoryRef> Bpart(layout.begin() + meta_.K * 2, layout.begin() + meta_.K * 4);
    return Bpart;
  }

  std::vector<MemoryRef> &in(int it) { return inputs_[it]; }
  std::vector<MemoryRef> &out(int it) { return outputs_[it];}
  size_t size() {return inputs_.size(); }
};

} // namespace gpuio::execution::ops::join
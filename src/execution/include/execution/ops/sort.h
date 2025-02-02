#pragma once

namespace gpuio::execution::ops::sort {

template <typename T>
std::pair<size_t, size_t> valBound(const T *ptr, size_t beg, size_t end, T v) {
  auto b = ptr + beg, e = ptr + end;
  auto lb = std::lower_bound(b, e, v);
  auto ub = std::upper_bound(b, e, v);
  return {lb - ptr , ub - ptr};
}

struct BoundsInfo {
  std::vector<std::pair<size_t, size_t>> bounds;
  size_t max, min;
};

template <typename T>
BoundsInfo valBounds(const T *ptr, size_t N, size_t chunk_size, T v) {
  std::vector<std::pair<size_t, size_t>> bounds;
  size_t mi = 0, ma = 0;
  for (size_t beg = 0; beg < N; beg += chunk_size) {
    size_t end = std::min(N, beg + chunk_size);
    auto r = valBound(ptr, beg, end, v);
    bounds.push_back(r);
    mi += (r.first - beg);
    ma += (r.second - beg);
  }

  return {bounds, ma, mi};
}

inline std::vector<size_t> radixSearchPivots(const uint64_t *ptr, size_t N, size_t chunk_size, size_t cnt) {
  uint64_t v = 0;
  for (int i = 63; i >= 0; i--) {
    uint64_t sv = v | (static_cast<uint64_t>(1) << i);
    // fmt::print("i: {}, sv: {}\n", i, sv);
    auto bi = valBounds(ptr, N, chunk_size, sv);
    // fmt::print("min: {}, max: {}\n", bi.min, bi.max);
    assert(bi.min == bi.max - 1 || bi.min == bi.max); // assume this is true for now, not so related to the key idea
    if (bi.max <= cnt) {
      v = sv;
    }
    // fmt::print("v: {}\n", sv);
  }

  auto fbi = valBounds(ptr, N, chunk_size, v);
  std::vector<size_t> r;
  for (auto &b: fbi.bounds) {
    r.push_back(b.first);
  }
  return r;
}

inline std::vector<std::vector<size_t>> groupForMerge(const uint64_t *ptr, size_t N, size_t chunk_size, size_t group_size) {
  size_t numInChunks = (N + chunk_size - 1) / chunk_size;
  std::vector<std::vector<size_t>> res;
  for (size_t i = 0; i < numInChunks; i++) {
    res.push_back({i * chunk_size});
  }

  for (size_t cnt = group_size; cnt < N; cnt += group_size) {
    auto ps = radixSearchPivots(ptr, N, chunk_size, cnt);
    for (size_t i = 0; i < numInChunks; i++) {
      res[i].push_back(ps[i]);
    }
  }

  for (size_t i = 0; i < numInChunks; i++) {
    res[i].push_back((i + 1) * chunk_size);
  }

  return res;
}

template <typename T>
struct MergeLevelsOp {
  size_t bufSize_;
  std::vector<std::vector<MemoryRef>> inputs_, outputs_;
  std::vector<std::vector<size_t>> divs_;

  void *temp_ptr_;
  size_t temp_size_;

  MergeLevelsOp(void *temp_ptr, size_t temp_size, MemoryRef hostSrc, MemoryRef hostDst, size_t chunk_size, std::vector<std::vector<size_t>> &groups) 
    : bufSize_(chunk_size * sizeof(T)), temp_ptr_(temp_ptr), temp_size_(temp_size) {

    assert(hostSrc.size % chunk_size == 0);

    size_t numChunks = hostSrc.size / (chunk_size * sizeof(T));

    for (int i = 0; i < numChunks; i++) {
      std::vector<MemoryRef> in;
      std::vector<size_t> div;
      size_t acc = 0;
      div.push_back(acc);
      for (int j = 0; j < groups.size(); j++) {
        auto p = hostSrc.slice(groups[j][i] * sizeof(T), groups[j][i + 1] * sizeof(T));
        in.push_back(p);
        acc += p.size;
        div.push_back(acc);
      }
      assert(acc == chunk_size * sizeof(T));
      inputs_.push_back(in);
      divs_.push_back(div);
      outputs_.push_back({hostDst.slice(i * chunk_size * sizeof(T), (i + 1) * chunk_size * sizeof(T))});
    }
  }

  int operator()(MemoryRef mem, int type, int it, hipStream_t s) {
    double_buffer dbuf(mem.slice(0, bufSize_), mem.slice(bufSize_, bufSize_ + bufSize_), type);
    gpuio::kernels::sort::rocm::mergeLevels<T>(temp_ptr_, temp_size_, dbuf, divs_[it], s, true);
    return dbuf.cur_;
  }

  std::vector<MemoryRef> inBuf(MemoryRef mem, int type, int) { 
    double_buffer dbuf(mem.slice(0, bufSize_), mem.slice(bufSize_, bufSize_ + bufSize_), type);
    return {dbuf.current()}; 
  }
  std::vector<MemoryRef> outBuf(MemoryRef mem, int type, int) { 
    double_buffer dbuf(mem.slice(0, bufSize_), mem.slice(bufSize_, bufSize_ + bufSize_), type);
    return {dbuf.alternate()}; 
  }
  std::vector<MemoryRef> &in(int it) { return inputs_[it]; }
  std::vector<MemoryRef> &out(int it) { return outputs_[it]; }
  size_t size() { return inputs_.size(); }
};

template <typename T>
struct MergeOp {
  size_t bufSize_;
  std::vector<std::vector<MemoryRef>> inputs_, outputs_;

  void *temp_ptr_;
  size_t temp_size_;

  MergeOp(void *temp_ptr, size_t temp_size, MemoryRef hostSrc, MemoryRef hostDst, size_t chunk_size, int level) 
    : bufSize_(chunk_size * sizeof(T)), temp_ptr_(temp_ptr), temp_size_(temp_size) {
    size_t group = std::pow(2, level);
    auto AChunks = partitionMem(hostSrc, bufSize_ * group);
    auto ps = regroupSortedVectors<T>(AChunks, group * 2);
    size_t merged = 0;
    for (size_t i = 0; i < ps.size(); i += 2) {
      inputs_.emplace_back(std::vector<MemoryRef>{ps[i], ps[i + 1]});
      auto hostOut = hostDst.slice(merged, std::min(hostDst.size, merged + bufSize_));
      outputs_.emplace_back(std::vector<MemoryRef>{hostOut});
      merged += bufSize_;
    }
  }

  int operator()(MemoryRef mem, int, int it, hipStream_t s) {
    size_t mid = inputs_[it][0].size;
    MemoryRef in1 = mem.slice(0, mid), in2 = mem.slice(mid, bufSize_), out = mem.slice(bufSize_, bufSize_ + bufSize_);
    gpuio::kernels::sort::rocm::merge<T>(temp_ptr_, temp_size_, in1, in2, out, s, true);
    return 0;
  }

  std::vector<MemoryRef> inBuf(MemoryRef mem, int, int) { return {mem.slice(0, bufSize_)}; }
  std::vector<MemoryRef> outBuf(MemoryRef mem, int, int) { return {mem.slice(bufSize_, bufSize_ + bufSize_)}; }
  std::vector<MemoryRef> &in(int it) { return inputs_[it]; }
  std::vector<MemoryRef> &out(int it) { return outputs_[it]; }
  size_t size() { return inputs_.size(); }
};


template <typename T>
struct SortOp {
  size_t bufSize_;
  std::vector<std::vector<MemoryRef>> inputs_, outputs_;

  void *temp_ptr_;
  size_t temp_size_;

  SortOp(void *temp_ptr, size_t temp_size, MemoryRef hostSrc, MemoryRef hostDst, size_t chunk_size) 
    : bufSize_(chunk_size * sizeof(T)), temp_ptr_(temp_ptr), temp_size_(temp_size) {
    auto srcChunks = partitionMem(hostSrc, bufSize_);
    auto dstChunks = partitionMem(hostDst, bufSize_);

    for (auto c: srcChunks) {
      inputs_.emplace_back(std::vector<MemoryRef>{c});
    }
    for (auto c: dstChunks) {
      outputs_.emplace_back(std::vector<MemoryRef>{c});
    }
  }

  int operator()(MemoryRef mem, int type, int it, hipStream_t s) {
    double_buffer dbuf(mem.slice(0, bufSize_), mem.slice(bufSize_, bufSize_ + bufSize_), type);
    dbuf.swap();
    gpuio::kernels::sort::rocm::radix_sort_keys<T>(temp_ptr_, temp_size_, dbuf, s);
    return dbuf.cur_;
  }

  std::vector<MemoryRef> inBuf(MemoryRef mem, int type, int) {
    double_buffer dbuf(mem.slice(0, bufSize_), mem.slice(bufSize_, bufSize_ + bufSize_), type);
    return {dbuf.alternate()};
  }

  std::vector<MemoryRef> outBuf(MemoryRef mem, int type, int) {
    double_buffer dbuf(mem.slice(0, bufSize_), mem.slice(bufSize_, bufSize_ + bufSize_), type);
    return {dbuf.current()};
  }

  std::vector<MemoryRef> &in(int it) { return inputs_[it]; }
  std::vector<MemoryRef> &out(int it) { return outputs_[it]; }
  size_t size() { return inputs_.size(); }
};

} // namespace gpuio::execution::ops::sort
#pragma once

#include <hipWrapper.h>
#include <io-sched.h>
#include <sort/sort.h>
#include <join/join.h>

/*
 * An execution layer built upon the IO primitive executes on GPU kernels.
 */

namespace gpuio::execution {

using gpuio::hip::MemoryRef;
using gpuio::kernels::double_buffer;

struct LinearMemrefAllocator {
  MemoryRef mem;
  size_t cur = 0;

  LinearMemrefAllocator(MemoryRef _mem): mem(_mem) {}

  MemoryRef alloc(size_t n) {
    if (cur + n > mem.size) {
      throw std::runtime_error("LinearMemrefAllocator drained");
    }
    auto r = mem.slice(cur, cur + n);
    cur += n;
    return r;
  }

  size_t allocated() { return cur; }
  size_t free() { return mem.size - cur; }
};

struct dual_double_buffer_layout {
  double_buffer dbufs_[2];
  MemoryRef temp;
  int cur_ = 0;

  LinearMemrefAllocator allocator;

  dual_double_buffer_layout(MemoryRef mem, size_t inSize, size_t outSize) : allocator(mem) {
    MemoryRef in, out;
    in = allocator.alloc(inSize);
    out = allocator.alloc(outSize);
    dbufs_[0] = double_buffer{in, out};

    in = allocator.alloc(inSize);
    out = allocator.alloc(outSize);
    dbufs_[1] = double_buffer{in, out};

    temp = allocator.alloc(allocator.free());
  }

  double_buffer &cur() { return dbufs_[cur_]; }
  double_buffer &alt() { return dbufs_[1 - cur_];}
  void swap() { cur_ = 1 - cur_ ; }
};

using double_buffer_layout = dual_double_buffer_layout;

struct double_execution_layout {
  double_buffer bufs;
  MemoryRef temp;
  
  LinearMemrefAllocator allocator;

  double_execution_layout(MemoryRef mem, size_t size) : allocator(mem) {
    auto mem1 = allocator.alloc(size);
    auto mem2 = allocator.alloc(size);
    bufs = double_buffer{mem1, mem2};
    temp = allocator.alloc(allocator.free());
  }
};


template <typename T>
size_t partitionArray(const T *A, size_t Asz, const T *B, size_t Bsz, size_t target) {
  auto [Sp, Rp] = Asz <= Bsz ? std::make_tuple(A, B) : std::make_tuple(B, A);
  auto [Ssz, Rsz] = Asz <= Bsz ? std::make_tuple(Asz, Bsz) : std::make_tuple(Bsz, Asz);

  int64_t l = std::max(0l, static_cast<int64_t>(target) - static_cast<int64_t>(Rsz)), r = std::min(Ssz, target), smid, rmid;
  while (true) {
    smid = (l + r) / 2;
    rmid = target - smid;

    T sr = smid < Ssz ? Sp[smid] : std::numeric_limits<T>::max();
    T sl = smid > 0 ? Sp[smid - 1] : std::numeric_limits<T>::min();
    T rr = rmid < Rsz ? Rp[rmid] : std::numeric_limits<T>::max();
    T rl = rmid > 0? Rp[rmid - 1] : std::numeric_limits<T>::min();

    if (std::max(sl, rl) <= std::min(sr, rr)) {
      return Asz <= Bsz ? smid : rmid;
    }

    if (sr < rl) {
      l = smid + 1;
    } else if (rr < sl) {
      r = smid;
    }
  }
}

template <typename T>
std::vector<std::tuple<size_t, size_t>> evenlyDivideVectors(MemoryRef A, MemoryRef B, size_t pieces) {
  std::vector<std::tuple<size_t, size_t>> r;
  for (size_t i = 1; i < pieces; i++) {
    auto Ap = reinterpret_cast<const T*>(A.ptr);
    auto Asz = A.size / sizeof(T);
    auto Bp = reinterpret_cast<const T*>(B.ptr);
    auto Bsz = B.size / sizeof(T);
    size_t target = (Asz + Bsz) * i / pieces;


    size_t pA = partitionArray(Ap, Asz, Bp, Bsz, target), pB = target - pA;
    r.emplace_back(pA * sizeof(T), pB * sizeof(T));
  }
  return r;
}

template <typename T>
std::vector<MemoryRef> regroupSortedVectors(std::vector<MemoryRef> &in, size_t pieces) {
  assert(in.size() % 2 == 0);

  std::vector<MemoryRef> r;
  for (int i = 0; i < in.size(); i += 2) {
    auto div = evenlyDivideVectors<T>(in[i], in[i + 1], pieces);
    div.emplace_back(in[i].size, in[i + 1].size);
    for (int j = 0; j < div.size(); j++) {
      auto [enda, endb] = div[j];
      auto [bega, begb] = j == 0 ? std::tuple<size_t, size_t>{0, 0} : div[j - 1];
      r.push_back(in[i].slice(bega, enda));
      r.push_back(in[i + 1].slice(begb, endb));
    }
  }
  return r;
}

inline std::vector<MemoryRef> partitionMem(MemoryRef mem, size_t chunk_size) {
  std::vector<MemoryRef> res;
  size_t total = mem.size;
  for (size_t i = 0; i < total; i += chunk_size) {
    size_t end = std::min(total, i + chunk_size);
    res.push_back(mem.slice(i, end));
  }
  return res;
}

struct GeneralOP {
  // if data is transferred inside <mem> with <type> through corresponding inBuf reference,
  // function can be executed, which returns the <type> to interpert the result.
  // the <type> can be passed to outBuf to get the vector of memory reference that contains
  // the output
  virtual int operator()(MemoryRef mem, int, int it, hipStream_t s) = 0;
  virtual std::vector<MemoryRef> inBuf(MemoryRef mem, int, int it) = 0;
  virtual std::vector<MemoryRef> outBuf(MemoryRef mem, int, int it) = 0;
  virtual std::vector<MemoryRef> &in(int it) = 0; 
  virtual std::vector<MemoryRef> &out(int it) = 0;
  virtual size_t size() = 0;
};


template <typename OP>
struct StaticPipelineExecutor {
  OP op;
  double_buffer &buf;
  std::vector<int> inLayout{0, 0}, outLayout;


  template <typename... Args>
  StaticPipelineExecutor(double_execution_layout &layout, Args&&... args) : 
    op(layout.temp.ptr, layout.temp.size, std::forward<Args>(args)...), buf(layout.bufs) {}

  template <typename Exchange>
  void run(Exchange &exchange, gpuio::hip::Stream &s) {
    int outType;

    // prologue
    exchange.launch(op.inBuf(buf.alternate(), inLayout[0], 0), op.in(0), {}, {});
    exchange.sync();
    buf.swap();

    outType = op(buf.current(), inLayout[0], 0, s);
    exchange.launch(op.inBuf(buf.alternate(), inLayout[1], 1), op.in(1), {}, {});
    inLayout.push_back(outType);
    outLayout.push_back(outType);
    exchange.sync();
    s.synchronize();
    buf.swap();

    for (int it = 2; it < op.size(); it++) {
      outType = op(buf.current(), inLayout[it - 1], it - 1, s);
      exchange.launch(op.inBuf(buf.alternate(), inLayout[it], it), op.in(it), op.out(it - 2), op.outBuf(buf.alternate(), outLayout[it - 2], it - 2));
      inLayout.push_back(outType);
      outLayout.push_back(outType);
      exchange.sync();
      s.synchronize();
      buf.swap();
    }

    outType = op(buf.current(), inLayout[op.size() - 1], op.size() - 1, s);
    exchange.launch({}, {}, op.out(op.size() - 2), op.outBuf(buf.alternate(), outLayout[op.size() - 2], op.size() - 2));
    inLayout.push_back(outType);
    outLayout.push_back(outType);
    exchange.sync();
    s.synchronize();
    buf.swap();

    exchange.launch({}, {}, op.out(op.size() - 1), op.outBuf(buf.alternate(), outLayout[op.size() - 1], op.size() - 1));
    exchange.sync();
    buf.swap();
  }
};


} // namespace gpuio::execution
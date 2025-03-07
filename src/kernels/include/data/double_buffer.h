#pragma once

#include <hipWrapper.h>

namespace gpuio::kernels {

using gpuio::hip::MemoryRef;
struct double_buffer {
  MemoryRef bufs[2];
  int cur_ = 0;
  double_buffer() = default;
  double_buffer(MemoryRef current, MemoryRef althernate) {
    bufs[0] = current;
    bufs[1] = althernate;
  }
  double_buffer(MemoryRef current, MemoryRef althernate, int cur) {
    bufs[0] = current;
    bufs[1] = althernate;
    cur_ = cur;
  }

  MemoryRef current() const {
    return bufs[cur_];
  }

  MemoryRef alternate() const {
    return bufs[cur_ ^ 1];
  }

  void swap() {
    if (bufs[0].size == bufs[1].size) cur_ ^= 1;
  }
};

} // namespace gpuio::kernels
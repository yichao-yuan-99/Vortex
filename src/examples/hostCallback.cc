#include <fmt/core.h>
#include <hipWrapper.h>
#include <atomic>

struct A {
  std::atomic<int> i = 0;
  std::vector<gpuio::hip::CallbackTagWrapper> callbacks;
  hipStream_t s;

  void callback(int tag, hipStream_t s, hipError_t) {
    i.fetch_add(1);
    if (!done()) {
      fmt::print("{} {}\n", i.load(), tag);
      gpuio::hip::addHostCallback(s, callbacks[i % 5]);
    }
  }

  A(hipStream_t _s) : s(_s) {
    for (int i = 0; i < 5; i++) {
      using namespace std::placeholders;
      callbacks.emplace_back(i, std::bind(&A::callback, this, _1, _2, _3));
    }
  }

  void launch() {gpuio::hip::addHostCallback(s, callbacks[0]); }

  bool done() { return i >= 10; }
};

int main() {
  gpuio::hip::Stream stream;

  A a(stream);

  a.launch();

  while (!a.done()) continue;

}
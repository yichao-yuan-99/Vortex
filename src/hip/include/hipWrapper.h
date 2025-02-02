#pragma once
#include <hip/hip_runtime.h>
#include <iostream>
#include <functional>
#include <vector>

namespace gpuio::hip {
  /*
   * runtime API Wrappers
   */
  class MemoryRef;

  void HostMalloc(void **ptr, size_t size);
  void HostFree(void *ptr);
  void DeviceMalloc(void **ptr, size_t size);
  void DeviceFree(void *ptr);

  void MemcpyAsync(MemoryRef dst, MemoryRef src, hipStream_t s);
  void MemsetAsync(MemoryRef dst, int value, hipStream_t s);

  std::tuple<size_t, size_t> MemGetInfo();

  // --- Device
  int GetDevice();
  void SetDevice(int d);
  int GetDeviceCount();
  void DeviceEanblePeerAccess(int peer);
  bool DeviceCanAccessPeer(int device, int peer);
  hipDeviceProp_t GetDeviceProperties(int device);

  void DeviceSynchronize();

  // --- Stream
  hipStream_t StreamCreate();
  void StreamDestroy(hipStream_t s);
  bool StreamQuery(hipStream_t s);
  void StreamSynchronize(hipStream_t s);
  void StreamWaitEvent(hipStream_t s, hipEvent_t e);
  void StreamAddCallback(hipStream_t s, hipStreamCallback_t callback, void *userData);

  // --- Event
  hipEvent_t EventCreate();
  void EventDestroy(hipEvent_t e);
  void EventRecord(hipEvent_t e, hipStream_t s);
  void EventSynchronize(hipEvent_t e);
  bool EventQuery(hipEvent_t e);
  float EventElapsedTime(hipEvent_t start, hipEvent_t stop);

  // --- Kernel
  template <typename... Args, typename F = void (*)(Args...)>
  void LanuchKernel(F&& kernel, const dim3& numBlocks, const dim3& dimBlocks,
      uint32_t sharedMemBytes, hipStream_t s, Args&&... args) {
    hipLaunchKernelGGL(std::forward<F>(kernel), numBlocks, dimBlocks, sharedMemBytes, s, std::forward<Args>(args)...);
    hipError_t launchError = hipGetLastError();
    if (launchError != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(launchError) << std::endl;
    }
  }

  /*
   * Control Primitives Proxy
   */

  class DeviceGuard {
  private:
    int original_device_id;
    int current_device_id;

  public:
    explicit DeviceGuard(int id);
    DeviceGuard(const DeviceGuard &) = delete;
    DeviceGuard &operator=(const DeviceGuard &) = delete;
    ~DeviceGuard();

    int device() const { return current_device_id; }
  };

  class Stream {
    hipStream_t stream_;
    int device_ = -1;
  public:
    using handle_t = hipStream_t;
    Stream();
    Stream(Stream &&s);
    ~Stream();

    operator hipStream_t() { return stream_; }
    int device() { return device_; }

    bool query() { return StreamQuery(stream_); }
    void synchronize() {return StreamSynchronize(stream_); }
    void waitEvent(hipEvent_t e) { StreamWaitEvent(stream_, e); }
  };

  class Event {
    hipEvent_t e_;
    int device_ = -1;
  public:
    Event();
    Event(Event &&e);
    ~Event();

    operator hipEvent_t() { return e_; }
    int device() { return device_; }

    void record(hipStream_t s) { EventRecord(e_, s); }
    void synchronize() {EventSynchronize(e_); }
    bool query() { return EventQuery(e_); }
  };

  inline float ElapsedTime(Event &start, Event &stop) {
    return EventElapsedTime(start, stop);
  }

  /*
   * Memory Primitive Proxy
   */

  template<typename T>
  struct HostAllocator {
    using value_type = T;

    HostAllocator() noexcept {}

    HostAllocator(const HostAllocator<T>&) noexcept {}

    T* allocate(std::size_t n) {
      T* ptr;
      HostMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T));
      return ptr;
    }

    void deallocate(T* p, std::size_t) noexcept {
      HostFree(p);
    }
  };

  template <typename T>
  using HostVector = std::vector<T, HostAllocator<T>>;

  struct DeviceMemoryDeleter {
    void operator()(void* ptr) const {
      DeviceFree(ptr);
    }
  };

  class DeviceMemory {
    std::unique_ptr<void, DeviceMemoryDeleter> ptr_{nullptr};
    size_t size_{0};
    int device_{-1};
  public:
    DeviceMemory() = default;
    DeviceMemory(size_t size);

    void *get() {return ptr_.get(); }
    const void *get() const {return ptr_.get(); }

    template <typename T>
    operator T*() { return reinterpret_cast<T *>(get()); }
    template <typename T>
    operator const T*() const { return reinterpret_cast<T *>(get()); }

    size_t size() const { return size_; }
    int device() const {return device_; }
  };

  struct MemoryRef {
    uint8_t *ptr = nullptr;
    size_t size = 0;;
    int device = -1;

  private:
    MemoryRef(uint8_t *_ptr, size_t _size, int _device) : ptr(_ptr), size(_size), device(_device) {}

  public:
    MemoryRef() = default;
    MemoryRef(int d): device(d) {}
    template<typename T>
    MemoryRef(HostVector<T> &rhs) : 
      ptr(reinterpret_cast<uint8_t *>(&rhs[0])), size(rhs.size() * sizeof(T)), device(-1) {}

    MemoryRef(DeviceMemory &rhs) :
      ptr(reinterpret_cast<uint8_t *>(rhs.get())), size(rhs.size()), device(rhs.device()) {}

    bool onHost() {return device == -1; }
    bool onDevice() {return !onHost(); }
    MemoryRef slice(size_t beg, size_t end) {return MemoryRef{ptr + beg, end - beg, device}; }
    MemoryRef slice_n(size_t beg, size_t bytes) {return slice(beg, beg + bytes); }

    template <typename T>
    operator T*() { return reinterpret_cast<T *>(ptr); }
    template <typename T>
    operator const T*() const { return reinterpret_cast<T *>(ptr); }
  };


  template <typename T>
  MemoryRef slice_n(MemoryRef mem, size_t beg, size_t num) {
    return mem.slice(std::min(beg * sizeof(T), mem.size), std::min((beg + num) * sizeof(T), mem.size));
  }

  template <typename T>
  MemoryRef slice_n(HostVector<T> &vec, size_t beg, size_t num) {
    return slice_n<T>(MemoryRef{vec}, beg, num);
  }

  template <typename T>
  MemoryRef slice(HostVector<T> &vec, size_t beg, size_t end) {
    return slice_n(vec, beg, end - beg);
  }

  /*
   * Host CallBack 
   */
  template<typename T, void (T::*f)(hipStream_t, hipError_t)>
  void HostCallback(hipStream_t s, hipError_t err, void *data) {
    std::invoke(f, static_cast<T *>(data), s , err);
  }

  template<typename T, void (T::*f)(hipStream_t, hipError_t) = &T::operator()>
  void addHostCallback(hipStream_t s, T &obj) {
    T *obj_ptr = &obj;
    void *data = static_cast<void *>(obj_ptr);
    auto c = &HostCallback<T, f>;
    StreamAddCallback(s, c, data);
  }

  template<typename T, void (T::*f)(hipStream_t, hipError_t) = &T::operator()>
  void addHostCallback(hipStream_t s, T *obj_ptr) {
    addHostCallback<T, f>(s, *obj_ptr);
  }

  struct CallbackTagWrapper {
    using callback_t = std::function<void (int, hipStream_t, hipError_t)>;
    const int tag;
    callback_t func;
    CallbackTagWrapper(int _tag, callback_t _func): tag(_tag), func(_func) {};

    void operator()(hipStream_t s, hipError_t e) {
      std::invoke(func, tag, s, e);
    }
  };

  /*
   * Platform Information
   */
  class Platform {
    int deviceCount_ = 0;
    std::vector<hipDeviceProp_t> deviceProps_;

    void EnableAllPeerAccess();  
    void warmUp();
  public:
    Platform();

    int deviceCount() { return deviceCount_; }
    hipDeviceProp_t deviceProp(int device) { return deviceProps_[device]; }
  };

  extern Platform platform;
}


/*
 * Utilities
 */

namespace gpuio::utils {

namespace units {
  inline double bytesToKB(size_t s) { return (double) s / 1024.0; }
  inline double bytesToGB(size_t s) { return (double) s / 1024.0 / 1024.0 / 1024.0; }
} // namespace units

namespace rand {
  void fill_rand_uint32(gpuio::hip::MemoryRef mem, uint64_t seed);
  void fill_rand_uint64(gpuio::hip::MemoryRef mem, uint64_t seed);

  template <typename T>
  void fill_rand(gpuio::hip::MemoryRef mem, uint64_t seed);

  template <typename T>
  void fill_rand_host(gpuio::hip::HostVector<T> &vec, gpuio::hip::MemoryRef mem, uint64_t seed);
} // namespace rand

namespace io {
  void loadBinary(gpuio::hip::MemoryRef mem, const std::string &file);
  void writeBinary(gpuio::hip::MemoryRef mem, const std::string &file);
} // namespace io

namespace time {
  double timeit(std::function<void()> f); 
} // namespace time

} // namespace gpuio::utils


/*
 * Random stuff 
 */

namespace gpuio::misc {

  void printDeviceProp(hipDeviceProp_t prop);
  void printPeerAccessStatus(); 

} // namespace gpuio::misc

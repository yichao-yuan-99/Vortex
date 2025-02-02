#include <hipWrapper.h>
#include <fmt/core.h>

#include <rocrand/rocrand.h>
#include <iostream>
#include <fstream>

#define HIP_CHECK_ERROR(call) { \
    hipError_t error = call; \
    if (error != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(error); \
    } \
}

namespace gpuio::hip {
  /*
   * runtime API Wrappers
   */

  void HostMalloc(void **ptr, size_t size) {
    HIP_CHECK_ERROR(hipHostMalloc(ptr, size, hipHostMallocDefault));
  }

  void HostFree(void *ptr) { HIP_CHECK_ERROR(hipHostFree(ptr)); }

  void DeviceMalloc(void **ptr, size_t size) { HIP_CHECK_ERROR(hipMalloc(ptr, size)); }

  void DeviceFree(void *ptr) { HIP_CHECK_ERROR(hipFree(ptr)); }

  void MemcpyAsync(MemoryRef dst, MemoryRef src, hipStream_t s) {
    assert(dst.size == src.size);
    if (dst.device == -1 && src.device != -1) {
      HIP_CHECK_ERROR(hipMemcpyAsync(dst.ptr, src.ptr, src.size, hipMemcpyDeviceToHost, s));
    } else if (dst.device != -1 && src.device == -1) {
      HIP_CHECK_ERROR(hipMemcpyAsync(dst.ptr, src.ptr, src.size, hipMemcpyHostToDevice, s));
    } else if (dst.device != -1 && src.device != -1) {
      HIP_CHECK_ERROR(hipMemcpyPeerAsync(dst.ptr, dst.device, src.ptr, src.device, src.size, s))
    } else {
      assert(0);
    }
  }

  void MemsetAsync(MemoryRef dst, int value, hipStream_t s) {
    HIP_CHECK_ERROR(hipMemsetAsync(dst.ptr, value, dst.size, s));
  }

  std::tuple<size_t, size_t> MemGetInfo() {
    size_t left, total;
    HIP_CHECK_ERROR(hipMemGetInfo(&left, &total));
    return {left, total};
  }

  void DeviceSynchronize() {
    HIP_CHECK_ERROR(hipDeviceSynchronize());
  }

  // --- Device
  int GetDevice() {
    int r;
    HIP_CHECK_ERROR(hipGetDevice(&r));
    return r;
  }

  void SetDevice(int d) { HIP_CHECK_ERROR(hipSetDevice(d)); }

  int GetDeviceCount() {
    int c;
    HIP_CHECK_ERROR(hipGetDeviceCount(&c));
    return c;
  }

  void DeviceEnablePeerAccess(int peer) { HIP_CHECK_ERROR(hipDeviceEnablePeerAccess(peer, 0)); }

  bool DeviceCanAccessPeer(int device, int peer) {
    int r;
    HIP_CHECK_ERROR(hipDeviceCanAccessPeer(&r, device, peer));
    return r;
  }

  hipDeviceProp_t GetDeviceProperties(int device) {
    hipDeviceProp_t prop;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&prop, device));
    return prop;
  }

  // --- Stream
  hipStream_t StreamCreate() {
    hipStream_t r;
    HIP_CHECK_ERROR(hipStreamCreate(&r))
    return r;
  }

  void StreamDestroy(hipStream_t s) { HIP_CHECK_ERROR(hipStreamDestroy(s)); }

  bool StreamQuery(hipStream_t s) {
    hipError_t err = hipStreamQuery(s);
    if (err == hipSuccess)
      return true;
    else if (err == hipErrorNotReady)
      return false;
    else
      throw std::runtime_error("Failed to query stream");
  }

  void StreamSynchronize(hipStream_t s) {
    HIP_CHECK_ERROR(hipStreamSynchronize(s));
  }

  void StreamWaitEvent(hipStream_t s, hipEvent_t e) {
    HIP_CHECK_ERROR(hipStreamWaitEvent(s, e, 0));
  }

  void StreamAddCallback(hipStream_t s, hipStreamCallback_t callback, void *userData) {
    HIP_CHECK_ERROR(hipStreamAddCallback(s, callback, userData, 0));
  }

  // --- Event
  hipEvent_t EventCreate() {
    hipEvent_t e;
    HIP_CHECK_ERROR(hipEventCreate(&e));
    return e;
  }

  void EventDestroy(hipEvent_t e) { HIP_CHECK_ERROR(hipEventDestroy(e)); }

  void EventRecord(hipEvent_t e, hipStream_t s) { 
    HIP_CHECK_ERROR(hipEventRecord(e, s));
  }

  void EventSynchronize(hipEvent_t e) { HIP_CHECK_ERROR(hipEventSynchronize(e)); }

  bool EventQuery(hipEvent_t e) {
    hipError_t err = hipEventQuery(e);
    if (err == hipSuccess) 
      return true;
    else if (err == hipErrorNotReady) 
      return false;
    else 
      throw std::runtime_error("Failed to query Event");
  }

  float EventElapsedTime(hipEvent_t start, hipEvent_t stop) {
    float r;
    HIP_CHECK_ERROR(hipEventElapsedTime(&r, start, stop));
    return r;
  }


  /*
   * Control Primitives Proxy
   */

  Stream::Stream() : stream_(StreamCreate()), device_(GetDevice()) {}

  Stream::Stream(Stream &&s) {
    if (&s != this) {
      this->stream_ = s.stream_;
      this->device_ = s.device_;
      s.device_ = -1;
    }
  }

  Stream::~Stream() {
    if (device_ != -1) {
      StreamDestroy(stream_);
    }
  }

  Event::Event() : e_(EventCreate()), device_(GetDevice()) {}

  Event::Event(Event &&e) {
    if (&e != this) {
      this->e_ = e.e_;
      this->device_ = e.device_;
      e.device_ = -1;
    }
  }

  Event::~Event() {
    if (device_ != -1) {
      EventDestroy(e_);
    }
  }

  /*
   * Memory Primitive Proxy
   */

  DeviceGuard::DeviceGuard(int id) : original_device_id(GetDevice()), current_device_id(id) {
    SetDevice(current_device_id);
  }

  DeviceGuard::~DeviceGuard() {
    SetDevice(original_device_id);
  }

  DeviceMemory::DeviceMemory(size_t size) : size_(size), device_(GetDevice()) {
    void *p;
    DeviceMalloc(&p, size_);
    ptr_.reset(p);
  }

  /*
   * Platform Information
   */
  void Platform::EnableAllPeerAccess() {
    for (int i = 0; i < deviceCount_; i++) {
      DeviceGuard on(i);
      for (int j = 0; j < deviceCount_; j++) {
        if (i == j) continue;
        DeviceEnablePeerAccess(j);
      }
    }
  }

  void Platform::warmUp() {
    std::vector<DeviceMemory> dev;
    std::vector<Stream> stream;
    for (int d = 0; d < deviceCount_; d++) {
      DeviceGuard on(d);
      dev.emplace_back(1000'000);
      stream.emplace_back();
    }
    HostVector<uint8_t> host(1000'000);
    for (int di = 0; di < deviceCount_; di++) {
      for (int dj = 0; dj < deviceCount_; dj++) {
        MemcpyAsync(dev[di], dev[dj], stream[di]);
      }
    }
    for (int d = 0; d < deviceCount_; d++) {
      MemcpyAsync(dev[d], host, stream[d]);
      MemcpyAsync(host, dev[d], stream[d]);
    }
    for (int d = 0; d < deviceCount_; d++) {
      stream[d].synchronize();
    }
  }
  
  Platform::Platform() : deviceCount_(GetDeviceCount()), deviceProps_(deviceCount_) {
    EnableAllPeerAccess();
    warmUp();
    for (int i = 0; i < deviceCount_; i++) {
      deviceProps_[i] = GetDeviceProperties(i);
    }
  }

  Platform platform;
}

/*
 * Utilities
 */

namespace gpuio::utils {

namespace rand {
  void fill_rand_uint32(gpuio::hip::MemoryRef mem, uint64_t seed) {
    assert(mem.onDevice());
    assert(mem.size % sizeof(uint32_t) == 0);

    auto size = mem.size / sizeof(uint32_t);
    auto ptr = reinterpret_cast<uint32_t *>(mem.ptr);

    rocrand_status r;
    rocrand_generator g;
    r = rocrand_create_generator(&g, ROCRAND_RNG_PSEUDO_THREEFRY4_32_20);
    if (r != ROCRAND_STATUS_SUCCESS) {
      throw std::runtime_error("rocrand cannot create generator");
    }

    r = rocrand_set_seed(g, seed);
    if (r != ROCRAND_STATUS_SUCCESS) {
      throw std::runtime_error("rocrand cannot set seed");
    }
    r = rocrand_generate(g, ptr, size);
    if (r != ROCRAND_STATUS_SUCCESS) {
      throw std::runtime_error("rocrand cannot generate");
    }

    r = rocrand_destroy_generator(g);
    if (r != ROCRAND_STATUS_SUCCESS) {
      throw std::runtime_error("rocrand cannot destroy generator");
    }
  }

  void fill_rand_uint64(gpuio::hip::MemoryRef mem, uint64_t seed) {
    assert(mem.onDevice());
    assert(mem.size % sizeof(uint64_t) == 0);
    assert(sizeof(uint64_t) == sizeof(unsigned long long));

    auto size = mem.size / sizeof(uint64_t);
    auto ptr = reinterpret_cast<unsigned long long *>(mem.ptr);

    rocrand_status r;
    rocrand_generator g;
    r = rocrand_create_generator(&g, ROCRAND_RNG_PSEUDO_THREEFRY4_64_20);
    if (r != ROCRAND_STATUS_SUCCESS) {
      throw std::runtime_error("rocrand cannot create generator");
    }

    r = rocrand_set_seed(g, seed);
    if (r != ROCRAND_STATUS_SUCCESS) {
      throw std::runtime_error("rocrand cannot set seed");
    }

    r = rocrand_generate_long_long(g, ptr, size);
    if (r != ROCRAND_STATUS_SUCCESS) {
      throw std::runtime_error("rocrand cannot generate");
    }

    r = rocrand_destroy_generator(g);
    if (r != ROCRAND_STATUS_SUCCESS) {
      throw std::runtime_error("rocrand cannot destroy generator");
    }
  }

  template <>
  void fill_rand<uint32_t>(gpuio::hip::MemoryRef mem, uint64_t seed) {
    fill_rand_uint32(mem, seed);
  }

  template <>
  void fill_rand<uint64_t>(gpuio::hip::MemoryRef mem, uint64_t seed) {
    fill_rand_uint64(mem, seed);
  }

  template <typename T>
  void fill_rand_host(gpuio::hip::HostVector<T> &vec, gpuio::hip::MemoryRef mem, uint64_t seed) {
    gpuio::hip::Stream s;
    size_t STEP = mem.size / sizeof(T);
    size_t total = vec.size();
    for (size_t i = 0; i < total; i += STEP) {
      size_t start = i;
      size_t end = std::min(i + STEP, total);
      size_t step = end - start;

      gpuio::hip::MemoryRef buf = mem.slice(0, sizeof(T) * step);
      auto hostRef = gpuio::hip::MemoryRef{vec}.slice(start * sizeof(T), end * sizeof(T));

      fill_rand<T>(buf, seed + (i / STEP));
      gpuio::hip::DeviceSynchronize();
      gpuio::hip::MemcpyAsync(hostRef, buf, s);
      s.synchronize();
    }
  }

  template void fill_rand_host<uint64_t>(gpuio::hip::HostVector<uint64_t> &vec, gpuio::hip::MemoryRef mem, uint64_t seed); 
  template void fill_rand_host<uint32_t>(gpuio::hip::HostVector<uint32_t> &vec, gpuio::hip::MemoryRef mem, uint64_t seed); 
} // namespace rand

namespace io {

  void loadBinary(gpuio::hip::MemoryRef mem, const std::string &file) {
    std::ifstream f(file, std::ios::in | std::ios::binary);
    if (!f.is_open()) {
      throw std::runtime_error("cannot open " + file);
    }
    f.read(reinterpret_cast<char *>(mem.ptr), mem.size);
    f.close();
  }

  void writeBinary(gpuio::hip::MemoryRef mem, const std::string &file) {
    std::ofstream f(file, std::ios::out | std::ios::binary);
    if (!f.is_open()) {
      throw std::runtime_error("cannot open " + file);
    }
    f.write(reinterpret_cast<char *>(mem.ptr), mem.size);
    f.close();
  }

} // namespace io

namespace time {
  double timeit(std::function<void()> f) {
    const auto start = std::chrono::high_resolution_clock::now();
    f();
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diff = end - start;
    return diff.count();
  }

} // namespace time

} // namespace gpuio::utils


/*
 * Random stuff 
 */

namespace gpuio::misc {

  void printDeviceProp(hipDeviceProp_t prop) {
    auto p = [](const std::string &s, auto v, const std::string &u = "") {
      fmt::print("{:<34}: {} {}\n", s, v, u);
    };

    fmt::print("{:-<34}\n", "");
    p("Name", prop.name);
    p("pciBusID", prop.pciBusID);
    p("pciDeviceID", prop.pciDeviceID);
    p("pciDomainID", prop.pciDomainID);
    p("multiProcessorCount", prop.multiProcessorCount);
    p("maxThreadsPerMultiProcessor", prop.maxThreadsPerMultiProcessor);
    p("isMultiGpuBoard", prop.isMultiGpuBoard);
    p("clockRate", prop.clockRate / 1000.0, "Mhz");
    p("memoryClockRate", prop.memoryClockRate / 1000.0, "Mhz");
    p("memoryBusWidth", prop.memoryBusWidth);
    p("totalGlobalMem", gpuio::utils::units::bytesToGB(prop.totalGlobalMem), "GiB");
    p("totalConstMem", prop.totalConstMem, "B");
    p("sharedMemPerBlock", prop.sharedMemPerBlock / 1024.0, "MiB");
    p("canMapHostMemory", prop.canMapHostMemory);
    p("regsPerBlock", prop.regsPerBlock);
    p("warpSize", prop.warpSize);
    p("l2CacheSize", prop.l2CacheSize);
    p("computeMode", prop.computeMode);
    p("maxThreadsPerBlock", prop.maxThreadsPerBlock);
    p("maxThreadsDim.x", prop.maxThreadsDim[0]);
    p("maxThreadsDim.y", prop.maxThreadsDim[1]);
    p("maxThreadsDim.z", prop.maxThreadsDim[2]);
    p("maxGridSize.x", prop.maxGridSize[0]);
    p("maxGridSize.y", prop.maxGridSize[1]);
    p("maxGridSize.z", prop.maxGridSize[2]);
    p("major", prop.major);
    p("minor", prop.minor);
    p("concunrrentKernels", prop.concurrentKernels);
    p("cooperativeLaunch", prop.cooperativeLaunch);
    p("cooperativeMultiDeviceLaunch", prop.cooperativeMultiDeviceLaunch);
    p("isIntegrated", prop.integrated);
    p("maxTexture1D", prop.maxTexture1D); 
    p("maxTexture2D.width", prop.maxTexture2D[0]);
    p("maxTexture2D.height", prop.maxTexture2D[1]);
    p("maxTexture3D.width", prop.maxTexture3D[0]);
    p("maxTexture3D.height", prop.maxTexture3D[1]);
    p("maxTexture3D.depth", prop.maxTexture3D[2]);
 
    #ifdef __HIP_PLATFORM_AMD__
    p("isLargeBar", prop.isLargeBar);
    p("asicRevision", prop.asicRevision);
    p("maxSharedMemoryPerMultiProcessor", gpuio::utils::units::bytesToKB(prop.maxSharedMemoryPerMultiProcessor), "KiB");
    p("clockInstructionRate", (float)prop.clockInstructionRate / 1000.0, "Mhz");
    p("arch.hasGlobalInt32Atomics", prop.arch.hasGlobalInt32Atomics);
    p("arch.hasGlobalFloatAtomicExch", prop.arch.hasGlobalFloatAtomicExch);
    p("arch.hasSharedInt32Atomics", prop.arch.hasSharedInt32Atomics);
    p("arch.hasSharedFloatAtomicExch", prop.arch.hasSharedFloatAtomicExch);
    p("arch.hasFloatAtomicAdd", prop.arch.hasFloatAtomicAdd);
    p("arch.hasGlobalInt64Atomics", prop.arch.hasGlobalInt64Atomics);
    p("arch.hasSharedInt64Atomics", prop.arch.hasSharedInt64Atomics);
    p("arch.hasDoubles", prop.arch.hasDoubles);
    p("arch.hasWarpVote", prop.arch.hasWarpVote);
    p("arch.hasWarpBallot", prop.arch.hasWarpBallot);
    p("arch.hasWarpShuffle", prop.arch.hasWarpShuffle);
    p("arch.hasFunnelShift", prop.arch.hasFunnelShift);
    p("arch.hasThreadFenceSystem", prop.arch.hasThreadFenceSystem);
    p("arch.hasSyncThreadsExt", prop.arch.hasSyncThreadsExt);
    p("arch.hasSurfaceFuncs", prop.arch.hasSurfaceFuncs);
    p("arch.has3dGrid", prop.arch.has3dGrid);
    p("arch.hasDynamicParallelism", prop.arch.hasDynamicParallelism);
    p("gcnArchName", prop.gcnArchName); 
    #endif
  }

  void printPeerAccessStatus() {
    int numDevice = gpuio::hip::platform.deviceCount();
    for (int i = 0; i < numDevice; i++) {
      std::vector<bool> c;
      for (int j = 0; j < numDevice; j++) {
        c.push_back(gpuio::hip::DeviceCanAccessPeer(i, j));
      }
      fmt::print("{:5}", bool(c[0]));
      for (int i = 1; i < numDevice; i++) {
        bool v = c[i];
        fmt::print(", {:5}", v);
      }
      fmt::print("\n");
    }
  }

} // namespace gpuio::misc
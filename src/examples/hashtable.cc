#include <kernels.h>
#include <hipWrapper.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <set>
#include <numeric>

template <typename T>
__global__ void read_kernel(const T *__restrict__ B,
                            const size_t N, uint64_t *A, bool secretlyFalse) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  double temp = B[tidx];

  if (secretlyFalse || temp == 123.0)
    A[tidx] = temp; 
}

int main() {
  using namespace gpuio;

  int tperSM = hip::platform.deviceProp(0).maxThreadsPerMultiProcessor;
  int tperB = hip::platform.deviceProp(0).maxThreadsPerBlock;
  int numSM = hip::platform.deviceProp(0).multiProcessorCount;

  int Bsize = std::gcd(tperSM, tperB);
  int numBperSM = tperSM / Bsize;
  int numB = numBperSM * numSM;

  fmt::print("{} blocks, {} threads each block\n", numB, Bsize);


  using Key_t = uint32_t;
  using Val_t = uint32_t;

  size_t BSIZE = 250'000'000; 
  hip::DeviceMemory Bkeys, Bvals;
  hip::HostVector<uint32_t> permutation(BSIZE);
  utils::io::loadBinary(permutation, "../data/permutation_1b_12138.bin");

  hip::HostVector<Key_t> Bkeys_h(permutation.begin(), permutation.end());
  hip::HostVector<Val_t> Bvals_h(BSIZE);
  std::iota(Bvals_h.begin(), Bvals_h.end(), 0);

  {
    hip::DeviceGuard on(0);
    Bkeys = hip::DeviceMemory(BSIZE * sizeof(Key_t));
    Bvals = hip::DeviceMemory(BSIZE * sizeof(Val_t));
    hip::Stream s;
    hip::MemcpyAsync(Bkeys, Bkeys_h, s);
    hip::MemcpyAsync(Bvals, Bvals_h, s);
    s.synchronize();
  }

  fmt::print("Bkeys: {} ... {}\n", 
    fmt::join(std::vector(&Bkeys_h[0], &Bkeys_h[6]), ", "),
    fmt::join(std::vector(&Bkeys_h[BSIZE - 5], &Bkeys_h[BSIZE]), ", ")
  );

  fmt::print("Bvals: {} ... {}\n", 
    fmt::join(std::vector(&Bvals_h[0], &Bvals_h[6]), ", "),
    fmt::join(std::vector(&Bvals_h[BSIZE - 5], &Bvals_h[BSIZE]), ", ")
  );

  using Filter = kernels::hashtable::Fraction<Key_t, 2>;
  using Hash = kernels::hashtable::hash::Identity<Key_t>;
  using HashMap_t = kernels::hashtable::HashMap<Key_t, Val_t, Hash>;
  using Pair_t = HashMap_t::Pair;

  // checksum
  uint64_t checksum = 0;
  for (size_t i = 0; i < Bkeys_h.size(); i++) {
    auto k = Bkeys_h[i];
    auto v = Bvals_h[i];
    if (Filter()(k)) {
      checksum += v;
    }
  }
  fmt::print("checksum: {}\n", checksum);

  gpuio::hip::DeviceMemory BHashMapMem;
  {
    hip::DeviceGuard on(0);
    hip::Stream s;
    BHashMapMem = hip::DeviceMemory(BSIZE * (sizeof(Key_t) + sizeof(Val_t)) * 2);
    hip::MemsetAsync(BHashMapMem, 0xff, s);
    s.synchronize();

    HashMap_t BHashMap(BHashMapMem);
    auto t = utils::time::timeit([&] {
      hip::LanuchKernel(kernels::hashtable::populateHashMap<Key_t, Val_t, Hash, Filter>, 
        dim3(numB), dim3(Bsize), 0, s, BHashMap, Bkeys, Bvals);
      s.synchronize();
    });
    fmt::print("build table time: {} s\n", t);

    hip::HostVector<Pair_t> BHashMap_h(BSIZE * 2);
    hip::MemcpyAsync(BHashMap_h, BHashMapMem, s);
    s.synchronize();

    hip::DeviceMemory tmp(128);
    hip::HostVector<uint64_t> tmp_h(128 / sizeof(uint64_t));

    uint64_t *out = reinterpret_cast<uint64_t *>(tmp.get());
    t = utils::time::timeit([&] {
      hip::LanuchKernel(kernels::hashtable::ops::probeAndSum<Key_t, Val_t, Hash>,
        dim3(numB), dim3(Bsize), 0, s, BHashMap, Bkeys, Bvals, out
      );
      s.synchronize();
    });
    fmt::print("proble and sum time: {} s\n", t);
    hip::MemcpyAsync(tmp_h, tmp, s);
    s.synchronize();
    fmt::print("result: {}, {}\n", tmp_h[0], tmp_h[0] == 2 * checksum);

    // t = utils::time::timeit([&] {
    //   hip::LanuchKernel(kernels::hashtable::ops::loadAndSum<Key_t>,
    //     dim3((BSIZE + 512 - 1) / 512), dim3(512), 0, s, 
    //     reinterpret_cast<uint64_t *>(Bvals.get()), BSIZE, out, false
    //   );
    //   s.synchronize();
    // });
    t = utils::time::timeit([&] {
      hip::LanuchKernel(read_kernel<Key_t>,
        dim3((BSIZE + 512 - 1) / 512), dim3(512), 0, s, 
        reinterpret_cast<Key_t *>(Bvals.get()), BSIZE, out, false
      );
      s.synchronize();
    });


    // t = utils::time::timeit([&] {
    //   hip::LanuchKernel(kernels::hashtable::ops::loadAndSum<Key_t>,
    //     dim3((BSIZE + 512 - 1) / 512), dim3(512), 0, s, 
    //     reinterpret_cast<uint64_t *>(Bvals.get()), BSIZE, out, false
    //   );
    //   s.synchronize();
    // });
    fmt::print("load and sum time: {} s\n", t);

  }


}
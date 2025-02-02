#include <hipWrapper.h>
#include <fmt/core.h>
#include <vector>

using namespace gpuio;

template <typename T>
__global__
void zero_copy_rand_pure(T *A, uint32_t *B, const T *__restrict__ Ah, const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  uint32_t id = B[tidx] % N;
  A[tidx] = Ah[id];
}

int main() {
  size_t N = 250'000'000;
  int blocksize = 512;
  int Nblock = (N + blocksize - 1) / blocksize;

  using dtype = uint64_t;

  hip::DeviceMemory A[4], B[4];
  dtype *Ap[4];
  uint32_t *Bp[4];
  hip::DeviceMemory C, D;
  std::vector<hip::Stream> ss;
  for (int d = 0; d < 4; d++) {
    hip::DeviceGuard on(d);
    A[d] = hip::DeviceMemory(sizeof(dtype) * N);
    B[d] = hip::DeviceMemory(sizeof(uint32_t) * N);

    Ap[d] = reinterpret_cast<dtype *>(A[d].get());
    Bp[d] = reinterpret_cast<uint32_t *>(B[d].get());
    ss.emplace_back();
  }
  {
    hip::DeviceGuard on(0);
    C = hip::DeviceMemory(sizeof(dtype) * 4 * N);
    D = hip::DeviceMemory(sizeof(uint32_t) * 4 * N);
  }
  dtype *Cp[4];
  for (int d = 0; d < 4; d++) {
    Cp[d] = reinterpret_cast<dtype *>(hip::slice_n<dtype>(C, d * N, N).ptr);
  }

  hip::HostVector<uint32_t> ids(N * 4);
  utils::io::loadBinary(ids, "../data/permutation_1b_12138.bin");
  for (int d = 0; d < 4; d++) {
    hip::MemcpyAsync(B[d], hip::slice_n<uint32_t>(ids, d * N, N), ss[d]);
    ss[d].synchronize();
  }
  hip::HostVector<dtype> Bh(ids.begin(), ids.end()), Ans(ids.size());
  for (size_t i = 0; i < Ans.size(); i++) {
    Ans[i] = Bh[ids[i] % N];
  }

  uint32_t *Dp[4]; // random ids on gpu0
  for (int d = 0; d < 4; d++) {
    Dp[d] = reinterpret_cast<uint32_t *>(hip::slice_n<uint32_t>(D, d * N, N).ptr);
  }
  hip::MemcpyAsync(D, ids, ss[0]);
  ss[0].synchronize();

  dtype *Ahp = reinterpret_cast<dtype *>(Bh.data());

  // for (int i = 0; i < 5; i++)
  // {
  //   fmt::print("all gpus to host\n");
  //   auto t = utils::time::timeit([&] {
  //     for (int d = 0; d < 4; d++) {
  //       hip::DeviceGuard on(d);
  //       hip::LanuchKernel(zero_copy_rand_pure<dtype>, dim3(Nblock), dim3(blocksize), 0, ss[d], Ap[d], Bp[d], Ahp, N);

  //     }
  //     for (int d = 0; d < 4; d++) {
  //       ss[d].synchronize();
  //     }
  //   });

  //   fmt::print("time: {}, IOPS: {}\n", t, N * 4 / t);
  // }

  for (int i = 0; i < 5; i++)
  {
    fmt::print("gather by all gpus\n");
    auto t = utils::time::timeit([&] {
      for (int d = 0; d < 4; d++) {
        hip::DeviceGuard on(d);
        hip::LanuchKernel(zero_copy_rand_pure<dtype>, dim3(Nblock), dim3(blocksize), 0, ss[d], Cp[d], Dp[d], Ahp, N);
      }
      for (int d = 0; d < 4; d++) {
        ss[d].synchronize();
      }
    });

    fmt::print("time: {}, IOPS: {}\n", t, N * 4 / t);
  }

  hip::MemcpyAsync(Bh, C, ss[0]);
  ss[0].synchronize();


  bool passed = true;
  for (int i = 0; i < Bh.size(); i++) {
    if (Bh[i] != Ans[i]) {
      fmt::print("{} {} != {}\n", i, Bh[i], Ans[i]);
      passed = false;
      break;
    }
  }

  fmt::print("passed: {}\n", passed);
  
}
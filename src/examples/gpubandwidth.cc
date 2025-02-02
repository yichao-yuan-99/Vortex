#include <hipWrapper.h>
#include <fmt/core.h>

template <typename T>
__global__ 
void read_kernel(T *A, const T *__restrict__ B,
                            const size_t N, bool secretlyFalse) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  double temp = B[tidx];

  if (secretlyFalse || temp == 123.0)
    A[0] = temp; 
}

template <typename T>
__global__
void rand_read(T *A, const T*__restrict__ B, const T*__restrict__ C, const size_t N, bool secretlyFalse, size_t r) {
  // __shared__ T V[1 * 1024];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;


  T id = B[tidx] % r;
  // V[threadIdx.x] = C[threadIdx.x];
  double temp = C[id];

  if (secretlyFalse || temp == 123.0)
    A[0] = temp; 
}

template <typename T>
__global__
void zero_copy_seq(T *B, const T *__restrict__ Bh, const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  B[tidx] = Bh[tidx];
}

template <typename T>
__global__
void zero_copy_rand(T *C, T *B, const T *__restrict__ Bh, const size_t N, size_t gran) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  T id = ((B[tidx / gran] % N) / gran ) * gran + tidx % gran;
  C[tidx] = Bh[id];
}

template <typename T>
__global__
void zero_copy_rand_pure(T *C, T *B, const T *__restrict__ Bh, const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  T id = B[tidx] % N;
  C[tidx] = Bh[id];
}

using namespace gpuio;

int main() {
{
  hip::DeviceGuard on(0);

  using dtype = uint32_t;

  // int N = 128ul * 1024 * 1024;
  size_t N = 250'000'000;
  int blocksize = 512;
  int Nblock = (N + blocksize - 1) / blocksize;

  hip::Stream s;
  hip::DeviceMemory A(128);
  hip::DeviceMemory B(N * sizeof(dtype));
  hip::DeviceMemory C(N * sizeof(dtype));

  hip::HostVector<uint32_t> permutation(N);
  utils::io::loadBinary(permutation, "../data/permutation_1b_12138.bin");
  hip::HostVector<dtype> Bh(permutation.begin(), permutation.end());

  hip::MemcpyAsync(B, Bh, s);
  s.synchronize();

  auto Bp = reinterpret_cast<dtype *>(B.get());
  auto Cp = reinterpret_cast<dtype *>(C.get());
  auto Ap = reinterpret_cast<dtype *>(A.get());

  auto Bhp = reinterpret_cast<dtype *>(Bh.data());

  auto t = utils::time::timeit([&] {

  hip::LanuchKernel(read_kernel<dtype>, 
    dim3(Nblock), dim3(blocksize), 0, s, Ap, Bp, N, false
  );
  s.synchronize();

  });

  hip::MemcpyAsync(B, Bh, s);
  s.synchronize();


  t = utils::time::timeit([&] {

  hip::LanuchKernel(read_kernel<dtype>, 
    dim3(Nblock), dim3(blocksize), 0, s, Ap, Bp, N, false
  );
  s.synchronize();

  });

  fmt::print("time: {} s, bw: {:.2f}\n", t, N * sizeof(dtype) / t / 1e9);

  t = utils::time::timeit([&] {

  hip::LanuchKernel(read_kernel<dtype>, 
    dim3(Nblock), dim3(blocksize), 0, s, Ap, Bp, N, false
  );
  s.synchronize();

  });

  fmt::print("time: {} s, bw: {:.2f}\n", t, N * sizeof(dtype) / t / 1e9);

  
  // for (int i = 512; i < 1024; i++) {
  //   auto t2 = utils::time::timeit([&] {

  //   hip::LanuchKernel(rand_read<dtype>, 
  //     dim3(Nblock), dim3(blocksize), 0, s, Ap, Bp, Cp, N, false, 1024 * i
  //   );
  //   s.synchronize();

  //   });

  //   fmt::print("{}, {}\n", i, t2);

  // }

  // for (int i = 2; i <= 128; i++) {
  //   auto t2 = utils::time::timeit([&] {

  //   hip::LanuchKernel(rand_read<dtype>, 
  //     dim3(Nblock), dim3(blocksize), 0, s, Ap, Bp, Cp, N, false, 1024 * 512 * i
  //   );
  //   s.synchronize();

  //   });

  //   fmt::print("{}, {}\n", 512 * i, t2);
  // }

  {
    auto t2 = utils::time::timeit([&] {
      hip::LanuchKernel(zero_copy_seq<dtype>, dim3(Nblock), dim3(blocksize), 0, s, Bp, Bhp, N);
      s.synchronize();
    });

    fmt::print("time: {}, bandwidth: {} GB/s\n", t2, N / 1e9 * sizeof(dtype) / t2);
  }
  
  std::vector<size_t> GS = {1, 2, 4, 8, 16, 32, 64};
  for (auto gran : GS)
  {
    // fmt::print("gran: {}\n", gran);
    auto t2 = utils::time::timeit([&] {
      hip::LanuchKernel(zero_copy_rand<dtype>, dim3(Nblock), dim3(blocksize), 0, s, Cp, Bp, Bhp, N, gran);
      s.synchronize();
    });

    fmt::print("{}, {}, {}\n", gran, t2, N / 1e9 * sizeof(dtype) / t2);
  }

  {
    auto t2 = utils::time::timeit([&] {
      hip::LanuchKernel(zero_copy_rand_pure<dtype>, dim3(Nblock), dim3(blocksize), 0, s, Cp, Bp, Bhp, N);
      s.synchronize();
    });

    fmt::print("time: {}, bandwidth: {} IOP/s\n", t2, N / t2);
  }
  
}
}

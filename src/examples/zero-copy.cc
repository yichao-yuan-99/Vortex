#include <zero-copy-multiple/data.h>
#include <fmt/core.h>

int main() {
  std::vector<gpuio::hip::Stream> ss;
  for (int d = 0; d < 4; d++) {
    gpuio::hip::DeviceGuard on(d);
    ss.emplace_back();
  }

  gpuio::hip::DeviceMemory mem0, mem1;
  gpuio::hip::HostVector<int> host(40 * 120);
  for (auto &v : host) v = -1;
  {
    gpuio::hip::DeviceGuard on(0);
    mem0 = gpuio::hip::DeviceMemory(40 * 120);
    gpuio::hip::MemsetAsync(mem0, 0xff, ss[0]);
    ss[0].synchronize();
  }
  {
    gpuio::hip::DeviceGuard on(1);
    mem1 = gpuio::hip::DeviceMemory(40 * 120);
    gpuio::hip::MemsetAsync(mem1, 0x0, ss[1]);
    ss[1].synchronize();
  }

  auto r = gpuio::utils::time::timeit([&] {
    gpuio::hip::DeviceGuard on(0);
    gpuio::hip::Stream s;
    volatile int *hh = host.data();
    gpuio::hip::LanuchKernel(gpuio::zcpy::pingpong<10, 0>, 1, 64, 0, ss[0], mem0, hh);
    // {
    //   gpuio::hip::DeviceGuard on(1);
    //   gpuio::hip::LanuchKernel(gpuio::zcpy::pingpong<10, 1>, 1, 64, 0, ss[1], mem1, mem0);
    // }
    for (int i = 0; i < 10; i++) {
      while(hh[0] != -1) continue;
      hh[0] = 0;
      gpuio::hip::MemsetAsync(gpuio::hip::MemoryRef{mem0}.slice(0, 4), 0xff, s);
      s.synchronize();
      fmt::print("host\n");
    }

    ss[0].synchronize();
    ss[1].synchronize();

  });
  fmt::print("time: {}\n", r);


  // auto r = gpuio::utils::time::timeit([&] {
  //   {
  //     gpuio::hip::DeviceGuard on(0);
  //     gpuio::hip::LanuchKernel(gpuio::zcpy::pingpong<10, 0>, 1, 64, 0, ss[0], mem0, mem1);
  //   }
  //   {
  //     gpuio::hip::DeviceGuard on(1);
  //     gpuio::hip::LanuchKernel(gpuio::zcpy::pingpong<10, 1>, 1, 64, 0, ss[1], mem1, mem0);
  //   }

  //   ss[0].synchronize();
  //   ss[1].synchronize();

  // });
  // fmt::print("time: {}\n", r);


  
  // // fill control block on host
  // gpuio::hip::HostVector<unsigned> freeListBlock(gpuio::zcpy::freeListBlockCnt());
  // gpuio::zcpy::fillInitialFreeListHost(freeListBlock);
  // for (int i = 0; i < 2; i++) {
  //   fmt::print("--------------\n");
  //   for (int j = 0; j < 34; j++) {
  //     fmt::print("{} ", freeListBlock[i * 34 + j]);
  //   }
  //   fmt::print("\n");
  // }


  // using dtype = uint32_t;

  // gpuio::hip::HostVector<uint32_t> host_vec(1'000'000'000);
  // gpuio::utils::io::loadBinary(host_vec, "../data/rand_uint32_4b.bin");

  // uint64_t hacc = 0;
  // for (auto v: host_vec) {
  //   hacc += v;
  // }
  // fmt::print("host acc: {}\n", hacc);

  // gpuio::hip::DeviceMemory mask, base_p, vals;
  // std::vector<gpuio::hip::Stream> ss;
  // for (int d = 0; d < 4; d++) {
  //   gpuio::hip::DeviceGuard on(d);
  //   ss.emplace_back();
  // }
  // // allocate device memory on GPU1
  // {
  //   gpuio::hip::DeviceGuard on(1);

  //   mask = gpuio::hip::DeviceMemory(gpuio::zcpy::ctrlBlockCnt() * sizeof(uint64_t));
  //   base_p = gpuio::hip::DeviceMemory(gpuio::zcpy::ctrlBlockCnt() * sizeof(uint64_t));
  //   vals = gpuio::hip::DeviceMemory(gpuio::zcpy::valBlockCnt() * sizeof(dtype));
  //   gpuio::hip::MemsetAsync(mask, 0, ss[1]);
  //   gpuio::hip::MemsetAsync(base_p, 0, ss[1]);
  //   ss[1].synchronize();
  // }
  
  // gpuio::hip::DeviceMemory acc, ctrls, comp;
  // {
  //   gpuio::hip::DeviceGuard on(0);

  //   ctrls = gpuio::hip::DeviceMemory(gpuio::zcpy::freeListBlockCnt() * sizeof(unsigned));
  //   comp = gpuio::hip::DeviceMemory(gpuio::zcpy::freeListBlockCnt() * sizeof(unsigned));
  //   acc = gpuio::hip::DeviceMemory(sizeof(uint64_t));
  //   gpuio::hip::MemsetAsync(acc, 0, ss[0]);
  //   gpuio::hip::MemsetAsync(ctrls, 0, ss[0]);
  //   gpuio::hip::MemsetAsync(comp, 0, ss[0]);
  //   ss[0].synchronize();
  // }

  // // setup forwarding service
  // {
  //   gpuio::hip::DeviceGuard on(1);
  //   auto numBlk = gpuio::zcpy::lanuchLoopingBlockNum<512>();
  //   gpuio::hip::LanuchKernel(gpuio::zcpy::WarpForwardingLoop<dtype>, dim3(numBlk), dim3(512), 0, ss[1], 
  //     mask, base_p, vals, comp
  //   );
  // }
  
  // // launch the kernel
  // auto r = gpuio::utils::time::timeit([&] {
  //   gpuio::hip::DeviceGuard on(0);
  //   auto numBlk = (1'000'000 + 255) / 256;
  //   dtype *host_v = host_vec.data();
  //   gpuio::hip::LanuchKernel(gpuio::zcpy::sumSelect<dtype>, numBlk, 256, 0, ss[0], 
  //     mask, base_p, vals, ctrls, comp, host_v, acc
  //   );
  //   ss[0].synchronize();
  // });

  // fmt::print("done {}\n", r);


  // // finishing

  // {
  //   gpuio::hip::DeviceGuard on(0);
  //   gpuio::hip::MemsetAsync(base_p, 0xff, ss[0]);
  //   ss[0].synchronize(); // looks like I cannot remove the synchronization here
  //   gpuio::hip::MemsetAsync(mask, 0xff, ss[0]);
  //   ss[0].synchronize();
  // }

  // {
  //   gpuio::hip::DeviceGuard on(1);
  //   ss[1].synchronize();

  // }

}
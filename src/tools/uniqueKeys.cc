#include <argparse/argparse.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <vector>
#include <algorithm>
#include <tbb/parallel_sort.h>
#include <random>
#include <fstream>

#include <hipWrapper.h>

// ./src/tools/uniqueKeys -N 4000000000 -s 12138 -ps 12138 -o ../data/uniqueKeys_uint64_4b_12138_12138.bin
// ./src/tools/uniqueKeys -N 4000000000 -s 12138 -ps 10086 -o ../data/uniqueKeys_uint64_4b_12138_10086.bin
// ./src/tools/uniqueKeys -N 2000000000 -s 12138 -ps 12138 -o ../data/uniqueKeys_uint64_2b_12138_12138.bin
// ./src/tools/uniqueKeys -N 2000000000 -s 12138 -ps 10086 -o ../data/uniqueKeys_uint64_2b_12138_10086.bin
// ./src/tools/uniqueKeys -N 6000000000 -s 12138 -ps 12138 -o ../data/uniqueKeys_uint64_6b_12138_12138.bin
// ./src/tools/uniqueKeys -N 6000000000 -s 12138 -ps 10086 -o ../data/uniqueKeys_uint64_6b_12138_10086.bin
int main(int argc, char **argv) {

  using dtype = uint64_t;

  argparse::ArgumentParser program("uniqueKeys");
  program.add_description("generate N unique uint64_t keys.");

  program.add_argument("-N").help("The number of keys to generate.").scan<'i', size_t>();
  program.add_argument("-s", "--seed").help("The seed to generate the unique keys.").scan<'i', size_t>();
  program.add_argument("-ps", "--permutation-seed").help("The seed to permutate the unique keys.").scan<'i', size_t>();
  program.add_argument("-o", "--output").help("The output file name.");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  size_t N = program.get<size_t>("-N");
  size_t seed = program.get<size_t>("-s");
  size_t pseed = program.get<size_t>("-ps");
  std::string output = program.get<std::string>("-o");
  

  fmt::print("N: {}, seed: {}, output: {}\n", N, seed, output);

  size_t toGen = N / 10 * 11;
  size_t chunkSize = 1000'000'000;
  fmt::print("generate {} random numbers...\n", toGen);

  gpuio::hip::DeviceGuard on(0);

  auto [left, total] = gpuio::hip::MemGetInfo();
  gpuio::hip::DeviceMemory big(left);
  gpuio::hip::MemoryRef bigMem = big;
  gpuio::hip::HostVector<dtype> keys(toGen);

  {
    auto randBuf = bigMem.slice(0, sizeof(dtype) * chunkSize);
    gpuio::utils::rand::fill_rand_host(keys, randBuf, seed);
    tbb::parallel_sort(keys);

    auto end = std::unique(keys.begin(), keys.end());
    size_t uniqueN = std::distance(keys.begin(), end);
    assert(uniqueN >= N);

    keys.resize(N);
    fmt::print("select the top {} keys...\n", N);

  }

  fmt::print("permute the generated numbers...\n");
  std::mt19937 g(pseed);
  std::shuffle(keys.begin(), keys.end(), g);

  fmt::print("result: {} ... {}\n", 
    fmt::join(std::vector<uint64_t>(keys.begin(), keys.begin() + 5), ", "),
    fmt::join(std::vector<uint64_t>(keys.begin() + N - 5, keys.begin() + N), ", ")
  );

  fmt::print("dump the result to {} ...\n", output);

  gpuio::utils::io::writeBinary(keys, output);

}
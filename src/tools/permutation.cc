#include <argparse/argparse.hpp>
#include <fmt/color.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>

// ./src/tools/permutation -m 1000000000 -s 12138 -o ../data/permutation_1b_12138.bin
// ./src/tools/permutation -m 4000000000 -s 12138 -o ../data/permutation_4b_12138.bin
int main(int argc, char **argv) {
  argparse::ArgumentParser program("permutation");
  program.add_description("generate a permutation of [0, max).");

  program.add_argument("-m", "--max").help("The max of index (at most max(uint32_t))").scan<'i', size_t>();
  program.add_argument("-s", "--seed").help("The seed to generate the permuation").scan<'i', size_t>(); 
  program.add_argument("-o", "--output").help("The output file name");

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  size_t max = program.get<size_t>("-m");
  size_t seed = program.get<size_t>("-s");
  std::string output = program.get<std::string>("-o");

  fmt::print("max: {}, seed: {}, output: {}\n", max, seed, output);

  std::mt19937 g(seed);
  std::vector<uint32_t> v(max);
  std::iota(v.begin(), v.end(), 0);
  
  fmt::print("before permutation: {}...\n", fmt::join(std::vector<uint32_t>(&v[0], &v[5]), ", "));

  std::shuffle(v.begin(), v.end(), g);
  fmt::print("after permutation: {}...\n", fmt::join(std::vector<uint32_t>(&v[0], &v[5]), ", "));
  fmt::print("dump the result to {} ...\n", output);

  std::ofstream f(output, std::ios::out | std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("cannot open " + output);
  }
  f.write(reinterpret_cast<char *>(&v[0]), v.size() * sizeof(uint32_t));
  f.close();
}
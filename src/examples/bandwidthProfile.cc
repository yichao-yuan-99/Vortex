#include <io-sched.h>

#include <fmt/ranges.h>

int main(int argc, char **argv) {
  if (argc != 3) {
    fmt::print("usage: ./bandwidthProfile <path-to-full-h2d> <path-to-full-d2h>\n");
    return -1;
  }

  std::string fullh2dFile = argv[1];
  std::string fulld2hFile = argv[1];

  gpuio::sched::profile::bandwidthProfile fullh2d(fullh2dFile, 19);
  gpuio::sched::profile::bandwidthProfile fulld2h(fulld2hFile, 19);

  fmt::print("{:-<50}\n", "");
  fmt::print("{}", fullh2d.get(0));
  fmt::print("{:-<50}\n", "");
  fmt::print("{}", fulld2h.get(0));
  fmt::print("{:-<50}\n", "");

  gpuio::sched::profile::patternOptions patternOptions(fullh2dFile, fulld2hFile);

  for (int i = 0; i < patternOptions.size(); i++) {
    fmt::print("{}\n", patternOptions.option(i));
  }
  fmt::print("{:-<50}\n", "");

  int beg = 1000'000, end = 2000'000;
  fmt::print("fit range {}\n", std::make_pair(beg, end));
  fmt::print("{:-<50}\n", "");
  for (int i = 0; i < patternOptions.size(); i++) {
    fmt::print("h2d: {}\n", patternOptions.option(i).fitH2D(beg, end));
    fmt::print("d2h: {}\n", patternOptions.option(i).fitD2H(beg, end));
    fmt::print("{:-<50}\n", "");
  }

}
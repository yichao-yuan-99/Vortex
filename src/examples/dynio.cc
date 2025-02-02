#include <io-sched.h>

#include <argparse/argparse.hpp>
#include <fmt/color.h>

using gpuio::hip::Stream;
using gpuio::hip::MemoryRef;
using gpuio::hip::DeviceMemory;
using gpuio::hip::HostVector;

struct ErrRecord {
  size_t totalTraffic, granularity, fraction;
  std::string msg;
};
std::vector<ErrRecord> errLog;

// return the result for 5 trails
std::vector<double> test(size_t totalTraffic, size_t granularity, size_t fraction, HostVector<uint32_t> &groundTruth, HostVector<uint32_t> &answer, int repeat) {
  double readRatio = (fraction * 1.0) / 10; 

  // the first element is readRatio
  std::vector<double> bandwidthResults = {readRatio};

  size_t H2DTraffic = totalTraffic / 10 * fraction, D2HTraffic = totalTraffic / 10 * (10 - fraction);
  fmt::print(fg(fmt::terminal_color::blue), "=== read ratio: {:>6.3f}, H2DTraffic: {:>12}, D2HTraffic: {:>12} ===\n", readRatio, H2DTraffic, D2HTraffic);
  MemoryRef H2DData = MemoryRef{groundTruth}.slice(0, H2DTraffic);
  MemoryRef D2HData = MemoryRef{groundTruth}.slice(H2DTraffic, totalTraffic);

  DeviceMemory dstDevice, srcDevice;
  {
    gpuio::hip::DeviceGuard on(0);
    dstDevice = DeviceMemory(H2DTraffic);
    srcDevice = DeviceMemory(D2HTraffic);
  }
  Stream setup;
  gpuio::hip::MemcpyAsync(srcDevice, D2HData, setup);
  setup.synchronize();

  uint32_t *H2DData_beg = reinterpret_cast<uint32_t *>(H2DData.ptr);
  uint32_t *H2DData_end = H2DData_beg + H2DData.size / sizeof(uint32_t);
  HostVector<uint32_t> srcHost(H2DData_beg, H2DData_end), dstHost(D2HTraffic / sizeof(uint32_t));

  for (int i = 0; i < repeat; i++) {
    // clean the environments
    std::fill(answer.begin(), answer.end(), 0);
    std::fill(dstHost.begin(), dstHost.end(), 0);
    gpuio::hip::MemsetAsync(dstDevice, 0, setup);
    setup.synchronize();


    // data transfer
    gpuio::sched::dyn::LoadBalancedExchange exchange(granularity);

    exchange.reset(dstDevice, srcHost, dstHost, srcDevice);
    auto t = gpuio::utils::time::timeit([&] {
      exchange.launch();
      exchange.sync();
    });
    auto bw = (H2DTraffic + D2HTraffic) / 1000'000'000.0/ t;
    bandwidthResults.push_back(bw);

    // get data from dsts
    MemoryRef answerH2D = MemoryRef{answer}.slice(0, H2DTraffic);
    gpuio::hip::MemcpyAsync(answerH2D, dstDevice, setup);
    setup.synchronize();
    std::copy(dstHost.begin(), dstHost.end(), answer.begin() + srcHost.size());

    // check answer
    bool correct = true;
    std::string errMsg;
    for (size_t i = 0; i < groundTruth.size(); i++) {
      if (groundTruth[i] != answer[i]) {
        correct = false;
        errMsg = fmt::format("At: {}, {} != {}", i, groundTruth[i], answer[i]);
        break;
      }
    }
    if (correct) {
      fmt::print(fg(fmt::terminal_color::green), "[Pass] ");
    } else {
      fmt::print(fg(fmt::terminal_color::red), "[Failed] {}\n", errMsg);
      errLog.emplace_back(ErrRecord{totalTraffic, granularity, fraction, errMsg});
    }
    fmt::print("time: {:8.4f} s, {:8.4f} GB/s\n", t, bw);
  }

  return bandwidthResults;
}

// ./dynio <totalTraffic> <granularity> <repeat>
int main(int argc, char **argv) {
  argparse::ArgumentParser program("dynio");

  program.add_argument("-t", "--totalTraffic")
    .help("the total amount of memory traffic")
    .default_value(size_t{16'000'000'000})
    .scan<'i', size_t>();

  program.add_argument("-g", "--granularity")
    .help("the granularity of transfer")
    .default_value(size_t{40'000'000})
    .scan<'i', size_t>();

  program.add_argument("-r", "--repeat")
    .help("# of experiments")
    .default_value(int{5})
    .scan<'i', int>();

  program.add_argument("-f", "--file")
    .help("the binary image used for the test")
    .default_value(std::string{"../data/rand_uint32_4b.bin"});
  
  program.add_description("test the bandwidth of dynamically scheduled IO.");

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  size_t totalTraffic = program.get<size_t>("-t");
  size_t granularity = program.get<size_t>("-g");
  int repeat = program.get<int>("-r");
  std::string groundtruthfilename = program.get<std::string>("-f");

  fmt::print(fg(fmt::terminal_color::yellow), "[test spec] total Traffic: {}, granularity: {}, repeat: {}\n", totalTraffic, granularity, repeat);

  // start doing the test

  HostVector<uint32_t> groundTruth(totalTraffic / sizeof(uint32_t));
  HostVector<uint32_t> answer(totalTraffic / sizeof(uint32_t));
  gpuio::utils::io::loadBinary(groundTruth, groundtruthfilename);


  std::vector<size_t> fractions = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  // std::vector<size_t> fractions = {5};

  std::vector<std::vector<double>> results;


  for (auto fraction : fractions) {
    auto r = test(totalTraffic, granularity, fraction, groundTruth, answer, repeat);
    results.push_back(r);
  }

  fmt::print("@@@ Result\n");
  for (auto &r: results) {
    fmt::print("{:8.4f}\n", fmt::join(r, ", "));
  }

  if (!errLog.empty()) {
    fmt::print("@@@ Errors\n");
    for (auto &e: errLog) {
      fmt::print("totalTraffic: {}, granularity: {}, fraction {}, Msg: {}\n", e.totalTraffic, e.granularity, e.fraction, e.msg);
    }
  }
}
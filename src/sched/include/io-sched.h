#pragma once

#include <hipWrapper.h>

#include <array>
#include <atomic>

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace gpuio::sched {
  using gpuio::hip::MemoryRef;
  using gpuio::hip::DeviceMemory;
  using gpuio::hip::Stream;
  using gpuio::hip::MemcpyAsync;

  class forwardingStage {
    using deviceBuf_t = std::vector<DeviceMemory>;
    using deviceStream_t = std::vector<Stream>;
    size_t size_;
    std::vector<deviceBuf_t> bufs_;
    std::vector<deviceStream_t> streams_;

  public:
    forwardingStage(size_t size);

    DeviceMemory &buffer(int device, int id) { return bufs_[device][id]; }
    // return the stream on dst (on src if dst is host)
    Stream &stream(int dst, int src);
    Stream &streamByBuf(int dst, int src, int bufdst, int bufsrc, int deviceId);

    size_t size() const { return size_; }
    void synchronize();
  };

} // namespace gpuio::sched

namespace gpuio::sched::profile {

extern std::array<int, 16> bitorderIndex;

class singleBandwidthProfile {
  std::array<double, 16 * 4> profile_;
public:
  singleBandwidthProfile(std::ifstream &f);

  double get(int code, int device) const {return profile_[4 * code + device]; }
};


class bandwidthProfile {
  std::vector<singleBandwidthProfile> profiles_;

public:
  bandwidthProfile(const std::string &file, size_t N);

  const singleBandwidthProfile &get(int id) { return profiles_[id]; }
};

using interval_t = std::pair<size_t, size_t>;
using taggedInterval_t = std::pair<int, interval_t>;
using taggedBandwidth_t = std::pair<int, double>;
using bandwidthRecord_t = std::vector<taggedBandwidth_t>;

struct patternOption {
  double rwRatio;
  bandwidthRecord_t h2d, d2h;

  struct comp {
    bool operator()(const patternOption &a, const patternOption &b);
  };

  patternOption(
    int h2dcode, const std::vector<double> &h2dbw, int d2hcode, const std::vector<double> &d2hbw
  );
  std::vector<taggedInterval_t> fitH2D(size_t beg, size_t end) const;
  std::vector<taggedInterval_t> fitD2H(size_t beg, size_t end) const;
private:
  bandwidthRecord_t cdf_(const bandwidthRecord_t &bw) const;
  std::vector<taggedInterval_t> fit_(const bandwidthRecord_t &cdf, size_t beg, size_t end) const;
};

class patternOptions {
  bandwidthProfile h2dfull, d2hfull;
  std::vector<patternOption> options_;

  bandwidthRecord_t toRecord(int code, const std::vector<double> &bw);

public:
  patternOptions(const std::string &h2dfullFile, const std::string &d2hfullFile);

  const std::vector<patternOption> &options() const { return options_; }
  const patternOption &option(int id) const {return options_[id]; }
  size_t size() const { return options_.size(); }
};


using MemoryID = std::tuple<int, int, size_t, size_t>; // device, local id, beg, end
using MemoryOp = std::pair<MemoryID, MemoryID>;
using ParallelOp = std::vector<MemoryOp>;
using Plan = std::vector<ParallelOp>;
using BindedMemoryOp = std::tuple<MemoryRef, MemoryRef, Stream *>;

/*
 * Each device has 4 buffers with size bufferSize
 */
struct PlanContext {
  MemoryRef dstDevice_, srcHost_, dstHost_, srcDevice_;
  forwardingStage &stage_;

  PlanContext(
    MemoryRef dstDevice, MemoryRef srcHost, MemoryRef dstHost, MemoryRef srcDevice, forwardingStage &stage
  );

  BindedMemoryOp bind(const MemoryOp &op);
  BindedMemoryOp bindByBuf(const MemoryOp &op);
private:
  MemoryRef bindMem(const MemoryID &mem);
};

class PlanGenerator {
public:
  friend struct fmt::formatter<PlanGenerator>;

  PlanGenerator(const PlanContext &context, const patternOptions &options);
  PlanGenerator(int deviceId, size_t H2DTraffic, size_t D2HTraffic, size_t bufferSize, const patternOptions &options);

  const Plan &plan() const { return plan_; }
  operator const Plan &() const { return plan(); }
  
private:
  const size_t bufferSize_;
  const patternOptions &options_;
  Plan plan_;
  int deviceId_;
  size_t H2DTraffic_, D2HTraffic_;
  double rwRatio_;
  std::vector<int> optionIds_;

  double mix_;
  int opIdLow_, opIdHigh_;

  // Begin with a two operation plan
  // pipeline the initial plan to multiple stages based on options
  Plan pipeline_();
  // take buffer into account, add memory operations to move the data to the boundary
  Plan complete_(const Plan& plan);
};


// a baseline solution ignoring all the pcie problems
class NaivePlanGenerator {
  Plan plan_;
  const size_t bufferSize_;
  int deviceId_;
  size_t H2DTraffic_, D2HTraffic_;

  // Begin with a two operation plan
  // pipeline the initial plan to multiple stages based on options
  Plan pipeline_();
  // take buffer into account, add memory operations to move the data to the boundary
  Plan complete_(const Plan& plan);

public:
  NaivePlanGenerator(const PlanContext &context);
  NaivePlanGenerator(int deviceId, size_t H2DTraffic, size_t D2HTraffic, size_t bufferSize);

  const Plan &plan() const { return plan_; }
  operator const Plan &() const { return plan(); }
};


class BlockPlanExecutor {
  PlanContext &context_;
  const Plan &plan_;

public:
  BlockPlanExecutor(PlanContext &context, const Plan &plan);

  void lanuch();
  void lanuch_s();
  void lanuch_ss();
  void lanuch_gg();
};


} // namespace gpuio::sched::profile

namespace gpuio::sched::dyn {

using gpuio::hip::Stream;
using gpuio::hip::MemoryRef;
using gpuio::hip::DeviceMemory;
using gpuio::hip::HostVector;

struct IOTask {
  MemoryRef dst, src;
  int id;
  bool done;
};

using nextFn_t = std::function<IOTask(int)>;
using reportFn_t = std::function<void(int, int)>;
using progressCallback_t = std::function<void(size_t)>;

class LoadBalancingSched {
  std::atomic<size_t> h2dCnt_{0}, d2hCnt_{0}, h2dIssued_{0}, d2hIssued_{0};
  size_t H2Dtraffic_{0}, D2Htraffic_{0}, gran_{0};
  std::array<std::atomic<int>, 10000> robh2d_ = {0};
  std::array<std::atomic<int>, 10000> robd2h_ = {0};

  std::vector<IOTask> h2dTasks_, d2hTasks_;

  std::thread h2dtrack_, d2htrack_;

  static void _h2dcallbackPrint(size_t finished) {}
  static void _d2hcallbackPrint(size_t finished) {}
  progressCallback_t h2dcallback_ = &LoadBalancingSched::_h2dcallbackPrint; 
  progressCallback_t d2hcallback_ = &LoadBalancingSched::_d2hcallbackPrint;

  void _produceTasks(std::vector<IOTask> &tasks, MemoryRef dst, MemoryRef src, size_t gran);

  int fixedD2H = 0;
public:
  LoadBalancingSched() = default;
  
  void reset(progressCallback_t h2dcallback, progressCallback_t d2hcallback) {
    h2dcallback_ = h2dcallback;
    d2hcallback_ = d2hcallback;
  }

  void reset(
    const std::vector<MemoryRef> &H2Ddsts, const std::vector<MemoryRef> &H2Dsrcs,
    const std::vector<MemoryRef> &D2Hdsts, const std::vector<MemoryRef> &D2Hsrcs,
    size_t gran
  );

  void reset(MemoryRef H2Ddst, MemoryRef H2Dsrc, MemoryRef D2Hdst, MemoryRef D2Hsrc, size_t gran);

  IOTask nextH2D(int); 
  IOTask nextD2H(int id); 
  IOTask next(int id); 
  void report(int id, int task); 

  operator nextFn_t() {
    using namespace std::placeholders;
    return std::bind(&LoadBalancingSched::next, this, _1);
  }

  operator reportFn_t() {
    using namespace std::placeholders;
    return std::bind(&LoadBalancingSched::report, this, _1, _2);
  }

  void h2dSent(); 
  void d2hSent(); 

  void launch() {
    h2dtrack_ = std::thread(&LoadBalancingSched::h2dSent, this);
    d2htrack_ = std::thread(&LoadBalancingSched::d2hSent, this);
  }

  void sync() {
    h2dtrack_.join();
    d2htrack_.join();
  }
};

struct IndirectLink {
  std::atomic<int> stage1Cnt_{0}, stage2Cnt_{0};
  int cur_{1}, next_{0};

  MemoryRef dst_, src_; 
  std::array<MemoryRef, 2> bufs_;
  MemoryRef bufDst_;
  int bufTaskId_;
  int lastSentId_;
  std::array<Stream *, 2> streams_;

  std::thread t_;
  nextFn_t nextFn_;
  reportFn_t reportFn_;

  int id_;

public:
  IndirectLink(MemoryRef buf1, MemoryRef buf2, Stream &stage1S, Stream &stage2S, nextFn_t nextFn, reportFn_t reportFn, int id)
   : bufs_({buf1, buf2}), streams_({&stage1S, &stage2S}), nextFn_(nextFn), reportFn_(reportFn), id_(id) {}

  void reset() {
    stage1Cnt_ = 0;
    stage2Cnt_ = 0;
    cur_ = 1;
    next_ = 0;
    bufDst_ = MemoryRef{};
    bufTaskId_ = -1;
    lastSentId_ = -1; 
  }

  void s1Callback(hipStream_t, hipError_t) { stage1Cnt_.fetch_add(1); }
  void s2Callback(hipStream_t, hipError_t) { stage2Cnt_.fetch_add(1); }
  void caller(); 
  void launch() { t_ = std::thread(&IndirectLink::caller, this); }
  void sync() { t_.join(); }
};

struct DirectLink {
  std::atomic<int> cnt_{0};
  Stream *s_ = nullptr;
  int lastSentId_;

  std::thread t_;
  nextFn_t nextFn_;
  reportFn_t reportFn_;

  int id_ = -1;
public:
  DirectLink(Stream &s, nextFn_t nextFn, reportFn_t reportFn, int id)
    : s_(&s), nextFn_(nextFn), reportFn_(reportFn), id_(id) {}


  void reset() {
    cnt_ = 0;
    lastSentId_ = -1;
  }

  void callback(hipStream_t, hipError_t) { cnt_.fetch_add(1); }
  void caller(); 
  void launch() { t_ = std::thread(&DirectLink::caller, this); }
  void sync() { t_.join(); }
};


void sleep_for(std::chrono::microseconds v); 

class ExchangeContextOwning {
  size_t gran_;
  std::vector<DeviceMemory> h2dbufs_, d2hbufs_;
  std::vector<Stream> h2dstreams_, d2hstreams_;

public:
  void _warmupfunc(hipStream_t, hipError_t) {}
  ExchangeContextOwning(size_t gran); 

  MemoryRef h2dbufs(int link, int id) { return h2dbufs_[link * 2 + id]; }
  MemoryRef d2hbufs(int link, int id) { return d2hbufs_[link * 2 + id]; }
  Stream &h2dS(int link, int id) { return h2dstreams_[link * 2 + id]; }
  Stream &d2hS(int link, int id) { return d2hstreams_[link * 2 + id]; }
  size_t size() const { return gran_; }
  void warmup(MemoryRef host); 
};


template <typename Sched>
class Exchange {
  ExchangeContextOwning cxt_;

  Sched sched_;

  std::array<DirectLink, 2> dlinks_;
  std::array<IndirectLink, 6> ilinks_;

  std::vector<MemoryRef> _divideRef(MemoryRef big, const std::vector<MemoryRef> &ps) {
    std::vector<MemoryRef> r;
    size_t cur = 0;
    for (auto p: ps) {
      r.push_back(big.slice(cur, cur + p.size));
      cur += p.size;
    }
    assert(cur == big.size);
    return r;
  }

public:
  Exchange(size_t gran) : cxt_(gran), 
  dlinks_{
    // h2d
    DirectLink{cxt_.h2dS(0, 0), sched_, sched_, 0},
    // d2h
    DirectLink{cxt_.d2hS(0, 0), sched_, sched_, 4 + 0},
  },
  ilinks_{
    // h2d
    IndirectLink{cxt_.h2dbufs(1, 0), cxt_.h2dbufs(1, 1), cxt_.h2dS(1, 0), cxt_.h2dS(1, 1), sched_, sched_, 1},
    IndirectLink{cxt_.h2dbufs(2, 0), cxt_.h2dbufs(2, 1), cxt_.h2dS(2, 0), cxt_.h2dS(2, 1), sched_, sched_, 2},
    IndirectLink{cxt_.h2dbufs(3, 0), cxt_.h2dbufs(3, 1), cxt_.h2dS(3, 0), cxt_.h2dS(3, 1), sched_, sched_, 3},
    // d2h
    IndirectLink{cxt_.d2hbufs(1, 0), cxt_.d2hbufs(1, 1), cxt_.d2hS(1, 0), cxt_.d2hS(1, 1), sched_, sched_, 4 + 1},
    IndirectLink{cxt_.d2hbufs(2, 0), cxt_.d2hbufs(2, 1), cxt_.d2hS(2, 0), cxt_.d2hS(2, 1), sched_, sched_, 4 + 2},
    IndirectLink{cxt_.d2hbufs(3, 0), cxt_.d2hbufs(3, 1), cxt_.d2hS(3, 0), cxt_.d2hS(3, 1), sched_, sched_, 4 + 3}
  }
  {
    HostVector<uint8_t> htmp_(cxt_.size());
    cxt_.warmup(htmp_);
  }


  void reset(MemoryRef dstDevice, MemoryRef srcHost, MemoryRef dstHost, MemoryRef srcDevice) {
    sched_.reset(dstDevice, srcHost, dstHost, srcDevice, cxt_.size());
    for (auto &dl: dlinks_) { dl.reset(); }
    for (auto &il: ilinks_) { il.reset(); }
  }

  void reset(
    const std::vector<MemoryRef> &H2Ddsts, 
    const std::vector<MemoryRef> &H2Dsrcs, 
    const std::vector<MemoryRef> &D2Hdsts, 
    const std::vector<MemoryRef> &D2Hsrcs 
  ) {
    const std::vector<MemoryRef> &H2Ddsts_ = (H2Ddsts.size() == 1 && H2Dsrcs.size() > 1) ? _divideRef(H2Ddsts[0], H2Dsrcs) : H2Ddsts;
    const std::vector<MemoryRef> &D2Hsrcs_ = (D2Hsrcs.size() == 1 && D2Hdsts.size() > 1) ? _divideRef(D2Hsrcs[0], D2Hdsts) : D2Hsrcs;
    sched_.reset(H2Ddsts_, H2Dsrcs, D2Hdsts, D2Hsrcs_, cxt_.size());
    for (auto &dl: dlinks_) { dl.reset(); }
    for (auto &il: ilinks_) { il.reset(); }
  }

  void launch() {
    sched_.launch();
    for (auto &dl: dlinks_) { dl.launch(); }
    for (auto &il: ilinks_) { il.launch(); }
  }

  template <typename... Args>
  void launch(Args&&... args) {
    reset(std::forward<Args>(args)...);
    launch();
  }

  void launch(const std::vector<MemoryRef> &H2Ddsts, const std::vector<MemoryRef> &H2Dsrcs, 
    const std::vector<MemoryRef> &D2Hdsts, const std::vector<MemoryRef> &D2Hsrcs 
  ) {
    reset(H2Ddsts, H2Dsrcs, D2Hdsts, D2Hsrcs);
    launch();
  }

  void sync() {
    sched_.sync();
    for (auto &dl: dlinks_) { dl.sync(); }
    for (auto &il: ilinks_) { il.sync(); }
  }
};

using LoadBalancedExchange = Exchange<LoadBalancingSched>;

} // namespace gpuio::sched::dyn

namespace gpuio::sched::naive {

struct NaiveExchange {
  std::vector<MemoryRef> _divideRef(MemoryRef big, const std::vector<MemoryRef> &ps) {
    std::vector<MemoryRef> r;
    size_t cur = 0;
    for (auto p: ps) {
      r.push_back(big.slice(cur, cur + p.size));
      cur += p.size;
    }
    assert(cur == big.size);
    return r;
  }

  std::vector<Stream> ss_;

  NaiveExchange() {}

  void launch(const std::vector<MemoryRef> &H2Ddsts, const std::vector<MemoryRef> &H2Dsrcs, 
    const std::vector<MemoryRef> &D2Hdsts, const std::vector<MemoryRef> &D2Hsrcs 
  ) {
    if (!(H2Ddsts.size() || D2Hsrcs.size())) return;
    // assert(H2Ddsts.size() || D2Hsrcs.size());
    int device = H2Ddsts.size() ? H2Ddsts[0].device : D2Hsrcs[0].device;
    gpuio::hip::DeviceGuard on(device);
    ss_.clear();
    ss_.emplace_back();
    ss_.emplace_back();
    // auto &s_ = ss_[0];
    // fmt::print("{} {} {} {}\n", H2Ddsts.size(), H2Dsrcs.size(), D2Hdsts.size(), D2Hsrcs.size());

    const std::vector<MemoryRef> &H2Ddsts_ = (H2Ddsts.size() == 1 && H2Dsrcs.size() > 1) ? _divideRef(H2Ddsts[0], H2Dsrcs) : H2Ddsts;
    const std::vector<MemoryRef> &D2Hsrcs_ = (D2Hsrcs.size() == 1 && D2Hdsts.size() > 1) ? _divideRef(D2Hsrcs[0], D2Hdsts) : D2Hsrcs;

    // fmt::print("{} {} {} {}\n", H2Ddsts_.size(), H2Dsrcs.size(), D2Hdsts.size(), D2Hsrcs_.size());
    size_t it = std::max(H2Dsrcs.size(), D2Hdsts.size());
    for (size_t i = 0; i < it; i++) {
      if (i < H2Dsrcs.size()) {
        gpuio::hip::MemcpyAsync(H2Ddsts_[i], H2Dsrcs[i], ss_[0]);
      }
      if (i < D2Hdsts.size()) {
        gpuio::hip::MemcpyAsync(D2Hdsts[i], D2Hsrcs_[i], ss_[1]);
      }
    }
    // fmt::print("done\n");
  }

  void sync() {
    if (ss_.size() == 0) return;
    // fmt::print("sync\n");
    ss_[0].synchronize();
    ss_[1].synchronize();
    // fmt::print("sync end\n");
  }
};


} // namespace gpuio::sched::naive

/*
 * libfmt formatters
 */
template <>
struct fmt::formatter<gpuio::sched::profile::singleBandwidthProfile> {
  auto parse(format_parse_context& ctx) { return ctx.begin(); }
  auto format(const gpuio::sched::profile::singleBandwidthProfile &v, format_context &ctx) const {
    for (int code = 0; code < 16; code ++) {
      for (int device = 0; device < 4; device++) {
        fmt::format_to(ctx.out(), "{:>8.3f} ", v.get(code, device));
      }
      fmt::format_to(ctx.out(), "\n");
    }
    return ctx.out();
  }
};

template <>
struct fmt::formatter<gpuio::sched::profile::patternOption> {
  auto parse(format_parse_context& ctx) { return ctx.begin(); }
  auto format (const gpuio::sched::profile::patternOption &v, format_context &ctx) const {
    char bufa[80], bufb[80];
    size_t l = 0; 
    for (int i = 0; i < v.h2d.size(); i++) {
      auto &t = v.h2d[i];
      if (60 - l <= 0) break;
      if (i != v.h2d.size() - 1) {
        auto r = fmt::format_to_n(&bufa[l], 60 - l,"({:1},{:3.1f}), ", t.first, t.second); 
        l += r.size;
      } else {
        auto r = fmt::format_to_n(&bufa[l], 60 - l,"({:1},{:3.1f})", t.first, t.second); 
        *r.out = '\0';
      }
    }
    l = 0;
    for (int i = 0; i < v.d2h.size(); i++) {
      auto &t = v.d2h[i];
      if (60 - l <= 0) break;
      if (i != v.d2h.size() - 1) {
        auto r = fmt::format_to_n(&bufb[l], 60 - l,"({:1},{:3.1f}), ", t.first, t.second); 
        l += r.size;
      } else {
        auto r = fmt::format_to_n(&bufb[l], 60 - l,"({:1},{:3.1f})", t.first, t.second); 
        *r.out = '\0';
      }
    }
    return fmt::format_to(ctx.out(), "rw: {:>4.3f}\n{:<40} | {:<40}", v.rwRatio, bufa, bufb);
  }
};

template <>
struct fmt::formatter<gpuio::sched::profile::Plan> {
  auto parse(format_parse_context& ctx) { return ctx.begin(); }
  auto format(const gpuio::sched::profile::Plan &v, format_context &ctx) const {
    char bufa[60], bufb[60];
    int step = 0;
    for (const auto &pops: v) {
      fmt::format_to(ctx.out(), "{:-<37} Step {:>3} {:-<37}\n", "", step, "");
      for (const auto &ops: pops) {
        auto &[a, b, c, d] = ops.second;
        auto pa = fmt::format_to_n(bufa, 33, "({:>2}, {:1}, {:>11}, {:>11})", a, b, c, d);
        *pa.out = '\0';
        auto &[e, f, g, h] = ops.first;
        auto pb = fmt::format_to_n(bufb, 33, "({:>2}, {:1}, {:>11}, {:>11})", e, f, g, h);
        *pb.out = '\0';

        fmt::format_to(ctx.out(), "{:33} => {:33} | {:>8.2f} MB\n", bufa, bufb, (d - c) / 1000.0 / 1000.0);
      }
      step++;
    }
    fmt::format_to(ctx.out(), "{:-<84}\n", "");
    return ctx.out();
  }
};

template <>
struct fmt::formatter<gpuio::sched::profile::PlanGenerator> {
  auto parse(format_parse_context& ctx) { return ctx.begin(); }
  auto format(const gpuio::sched::profile::PlanGenerator &v, format_context &ctx) const {
    fmt::format_to(ctx.out(), "{:-<37} MetaInfo {:-<37}\n", "", "");
    fmt::format_to(ctx.out(), "H2D: {:>8.2f} MB, D2H: {:>8.2f} MB, rw ratio: {:>4.3f}", v.H2DTraffic_ / (1000'000.0), v.D2HTraffic_ / (1000'000.0), v.rwRatio_);
    fmt::format_to(ctx.out(), ", opLow: {:>2}, opHigh: {:>2}\n", v.opIdLow_, v.opIdHigh_);
    fmt::format_to(ctx.out(), "{}\n", v.options_.option(v.opIdLow_));
    fmt::format_to(ctx.out(), "{}\n", v.options_.option(v.opIdHigh_));
    fmt::format_to(ctx.out(), "trafer granularity: {:>8.2f} MB, # of steps: {:>3}\n", v.bufferSize_ / (1000'000.0), v.plan_.size());
    fmt::format_to(ctx.out(), "{:>4.3f} low, {:>4.3f} high\n", v.mix_, 1 - v.mix_);
    fmt::format_to(ctx.out(), "selected opIds: {}\n", v.optionIds_);
    return fmt::format_to(ctx.out(), "{}", v.plan_);
  }

};
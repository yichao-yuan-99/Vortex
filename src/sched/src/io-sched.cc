#include <io-sched.h>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace gpuio::sched {

forwardingStage::forwardingStage(size_t size) : size_(size) {
  int devCount = gpuio::hip::platform.deviceCount();
  bufs_.resize(devCount);
  streams_.resize(devCount);

  for (int d = 0; d < devCount; d++) {
    gpuio::hip::DeviceGuard on(d);
    bufs_[d].emplace_back(size);
    bufs_[d].emplace_back(size);
    bufs_[d].emplace_back(size);
    bufs_[d].emplace_back(size);
    streams_[d].emplace_back();
    streams_[d].emplace_back();
    streams_[d].emplace_back();
    streams_[d].emplace_back();
    streams_[d].emplace_back();
  }
}

Stream &forwardingStage::stream(int dst, int src) {
  if (dst == -1) {
    return streams_[src].back();
  } else {
    if (src == -1) {
      return streams_[dst][dst];
    } else {
      return streams_[dst][src];
    }
  }
}

Stream &forwardingStage::streamByBuf(int dst, int src, int bufdst, int bufsrc, int deviceId) {
  if (dst == -1 && src == deviceId) {
    return streams_[deviceId][0];
  } else if (dst == deviceId && src == -1) {
    return streams_[deviceId][1];
  } else if (dst == -1 || dst == deviceId) {
    return streams_[src][bufsrc];
  } else {
    return streams_[dst][bufdst];
  }
}

void forwardingStage::synchronize() {
  for (auto &ss: streams_) {
    for (auto &s: ss) {
      s.synchronize();
    }
  }

}


} // namespace gpuio::sched

namespace gpuio::sched::profile {

std::array<int, 16> bitorderIndex = {0, 1, 2, 4, 8, 3, 5, 9, 6, 10, 12, 7, 11, 13, 14, 15};

singleBandwidthProfile::singleBandwidthProfile(std::ifstream &f){
  f.read(reinterpret_cast<char *>(profile_.data()), sizeof(double) * 16 * 4);
}

bandwidthProfile::bandwidthProfile(const std::string &file, size_t N) {
  std::ifstream f(file, std::ios::binary | std::ios::in);
  for (size_t i = 0; i < N; i++) {
    profiles_.emplace_back(f);
  }
}

patternOption::patternOption(
  int h2dcode, const std::vector<double> &h2dbw, int d2hcode, const std::vector<double> &d2hbw
) {
  double h2dtotalbw = 0.0, d2htotalbw = 0.0;
  for (int i = 0; i < 4; i++) {
    if (h2dcode & (1 << i)) {
      h2d.emplace_back(i, h2dbw[i]);
      h2dtotalbw += h2dbw[i];
    }
    if (d2hcode & (1 << i)) {
      d2h.emplace_back(i, d2hbw[i]);
      d2htotalbw += d2hbw[i];
    }
  }
  rwRatio = h2dtotalbw / (h2dtotalbw + d2htotalbw);
}

bool patternOption::comp::operator()(const patternOption &a, const patternOption &b) {
  return a.rwRatio < b.rwRatio;
}

std::vector<taggedInterval_t> patternOption::fitH2D(size_t beg, size_t end) const {
  auto cdf = cdf_(h2d);
  return fit_(cdf, beg, end);
}

std::vector<taggedInterval_t> patternOption::fitD2H(size_t beg, size_t end) const {
  auto cdf = cdf_(d2h);
  return fit_(cdf, beg, end);
}

bandwidthRecord_t patternOption::cdf_(const bandwidthRecord_t &bw) const {
  double sum = 0;
  for (const auto &v: bw) {
    sum += v.second;
  }

  bandwidthRecord_t r = bw;
  for (int i = 1; i < r.size(); i++) {
    r[i].second = r[i - 1].second + r[i].second;
  }
  for (auto &v: r) {
    v.second /= sum;
  }
  return r;
}

std::vector<taggedInterval_t> patternOption::fit_(const bandwidthRecord_t &cdf, size_t beg, size_t end) const {
  std::vector<taggedInterval_t> r;

  size_t l = end - beg;
  for (int i = 0; i < cdf.size(); i++) {
    double pa = i > 0 ? cdf[i - 1].second : 0.0;
    double pb = cdf[i].second;

    size_t f = beg + l * pa;
    size_t b = beg + l * pb;
    interval_t iv = {f, b};
    r.emplace_back(cdf[i].first, iv);
  }
  r.front().second.first = beg;
  r.back().second.second = end;

  return r;
}

bandwidthRecord_t patternOptions::toRecord(int code, const std::vector<double> &bw) {
  bandwidthRecord_t r;
  for (int i = 0; i < 4; i++) {
    if (code & (1 << i)) {
      r.emplace_back(i, bw[i]);
    }
  }
  return r;
}

patternOptions::patternOptions(const std::string &h2dfullFile, const std::string &d2hfullFile)
  : h2dfull(h2dfullFile, 19), d2hfull(d2hfullFile, 19) {

  // select some important bandwidth points
  // std::vector<int> h2dfullCode = {0b0001, 0b0011}; // d2h code
  // std::vector<int> d2hfullCode = {0b0001, 0b0011}; // h2d code
  std::vector<int> h2dfullCode = {0b0001}; // d2h code
  std::vector<int> d2hfullCode = {0b0001}; // h2d code
  std::vector<double> d2hBW = {24.0, 27.0, 27.0, 27.0}; // assume it is stable

  auto &h2dfull0 = h2dfull.get(0);
  for (auto &d2hcode: h2dfullCode) {
    std::vector<double> h2dbw;
    for (int d = 0; d < 4; d++) {
      h2dbw.push_back(h2dfull0.get(d2hcode, d));
    }
    options_.emplace_back(0b1111, h2dbw, d2hcode, d2hBW);
  }

  auto &d2hfull0 = d2hfull.get(0);
  for (auto &h2dcode: d2hfullCode) {
    std::vector<double> h2dbw;
    for (int d = 0; d < 4; d++) {
      h2dbw.push_back(d2hfull0.get(h2dcode, d));
    }
    options_.emplace_back(h2dcode, h2dbw, 0b1111, d2hBW);
  }

  std::sort(options_.begin(), options_.end(), patternOption::comp());
}

PlanContext::PlanContext(
  MemoryRef dstDevice, MemoryRef srcHost, MemoryRef dstHost, MemoryRef srcDevice, forwardingStage &stage
) : dstDevice_(dstDevice), srcHost_(srcHost), dstHost_(dstHost), srcDevice_(srcDevice), stage_(stage) {
  assert(dstHost_.device == -1 && srcHost_.device == -1 && dstDevice_.device == srcDevice_.device && dstDevice_.device != -1);
  fmt::print("{} {} {} {}\n", dstDevice_.size, srcHost_.size, dstHost_.size, srcDevice_.size);
  assert(dstDevice_.size == srcHost_.size && dstHost_.size == srcDevice_.size);
}

BindedMemoryOp PlanContext::bind(const MemoryOp &op) {
  auto dstRef = bindMem(op.first);
  auto srcRef = bindMem(op.second);

  auto &stream = stage_.stream(dstRef.device, srcRef.device);
  return {dstRef, srcRef, &stream};
}

BindedMemoryOp PlanContext::bindByBuf(const MemoryOp &op) {
  auto dstRef = bindMem(op.first);
  auto srcRef = bindMem(op.second);

  auto bufdst = std::get<1>(op.first);
  auto bufsrc = std::get<1>(op.second);

  auto &stream = stage_.streamByBuf(dstRef.device, srcRef.device, bufdst, bufsrc, dstDevice_.device);
  return {dstRef, srcRef, &stream};
}

PlanGenerator::PlanGenerator(const PlanContext &context, const patternOptions &options)
  : PlanGenerator(context.dstDevice_.device, context.dstDevice_.size, context.srcDevice_.size, context.stage_.size(), options) {}

PlanGenerator::PlanGenerator(int deviceId, size_t H2DTraffic, size_t D2HTraffic, size_t bufferSize, const patternOptions &options)
 : deviceId_(deviceId), H2DTraffic_(H2DTraffic), D2HTraffic_(D2HTraffic), bufferSize_(bufferSize), options_(options) {
  plan_ = pipeline_();
  // fmt::print("{}", plan_);
  plan_ = complete_(plan_);
  // fmt::print("{}", plan_);
}

Plan PlanGenerator::pipeline_() {
  size_t totalSize = H2DTraffic_ + D2HTraffic_;

  size_t num_chunk = (totalSize + bufferSize_ - 1) / bufferSize_;
  rwRatio_ = static_cast<double>(H2DTraffic_) / totalSize;
  // [FIXME] bug for skewed swap
  int pattern = 1;
  for (; pattern < options_.size(); pattern++) {
    if (rwRatio_ < options_.option(pattern).rwRatio) {
      break;
    }
  }

  opIdLow_ = pattern -1;
  opIdHigh_ = pattern;

  auto &ptLow = options_.option(pattern - 1);
  auto &ptHigh = options_.option(pattern);

  mix_ = (ptHigh.rwRatio - rwRatio_) / (ptHigh.rwRatio - ptLow.rwRatio);
  size_t nLow = num_chunk * mix_, nHigh = num_chunk - nLow;
  while (nLow || nHigh) {
    if (nLow) {
      nLow--;
      optionIds_.push_back(opIdLow_);
    }
    if (nHigh) {
      nHigh--;
      optionIds_.push_back(opIdHigh_);
    }
  }
  
  Plan result;
  size_t H2DSent = 0, D2HSent = 0;
  int cur = 1, next = 0;
  for (int i = 0; i < optionIds_.size(); i++) {
    auto &pt = options_.option(optionIds_[i]);
    size_t H2D = pt.rwRatio * bufferSize_, D2H = bufferSize_ - H2D;

    size_t H2DEnd = i == optionIds_.size() - 1 ? H2DTraffic_ : H2DSent + H2D;
    size_t D2HEnd = i == optionIds_.size() - 1 ? D2HTraffic_ : D2HSent + D2H;
    auto H2Dfit = pt.fitH2D(H2DSent, H2DEnd);
    auto D2Hfit = pt.fitD2H(D2HSent, D2HEnd);

    ParallelOp op;
    for (auto &h2dtask: H2Dfit) {
      int d = h2dtask.first;
      auto [beg, end] = h2dtask.second;
      if (d != deviceId_) {
        op.emplace_back(MemoryID{d, next, 0, end - beg}, MemoryID{-1, 0, beg, end});
      } else {
        op.emplace_back(MemoryID{d, 0, beg, end}, MemoryID{-1, 0, beg, end});
      }
    }
    for (auto &d2htask: D2Hfit) {
      int d = d2htask.first;
      auto [beg, end] = d2htask.second;
      if (d != deviceId_) {
        op.emplace_back(MemoryID{-1, 1, beg, end}, MemoryID{d, 2 + next, 0, end - beg});
      } else {
        op.emplace_back(MemoryID{-1, 1, beg, end}, MemoryID{d, 1, beg, end});
      }
    }
    H2DSent += H2D;
    D2HSent += D2H;

    result.push_back(op);

    std::swap(cur, next);
  }

  return result;
}

Plan PlanGenerator::complete_(const Plan &plan) {
  Plan result(plan.size() + 2);

  for (int step = 0; step < plan.size(); step++) {
    auto &parOps = plan[step];

    int i = step + 1;
    for (auto &op: parOps) {
      auto &[opDst, opSrc] = op;
      int opDstID = std::get<0>(opDst), opSrcID = std::get<0>(opSrc);

      if (opSrcID == -1) { // H2D
        if (opDstID != deviceId_) { // indirect
          size_t sendBeg, sendEnd;
          std::tie(std::ignore, std::ignore, sendBeg, sendEnd) = opSrc;
          result[i + 1].emplace_back(MemoryID{deviceId_, 0, sendBeg, sendEnd}, opDst);
        }
      } else { // D2H
        if (opSrcID != deviceId_) { // indirect
          size_t sendBeg, sendEnd;
          std::tie(std::ignore, std::ignore, sendBeg, sendEnd) = opDst;
          result[i - 1].emplace_back(opSrc, MemoryID{deviceId_, 1, sendBeg, sendEnd});
        }
      }
      result[i].push_back(op);
    }
  }
  return result;
}

NaivePlanGenerator::NaivePlanGenerator(const PlanContext &context)
  : NaivePlanGenerator(context.dstDevice_.device, context.dstDevice_.size, context.srcDevice_.size, context.stage_.size()) {}

NaivePlanGenerator::NaivePlanGenerator(int deviceId, size_t H2DTraffic, size_t D2HTraffic, size_t bufferSize) 
 : deviceId_(deviceId), H2DTraffic_(H2DTraffic), D2HTraffic_(D2HTraffic), bufferSize_(bufferSize) {
  plan_ = pipeline_();
  // fmt::print("{}", plan_);
  plan_ = complete_(plan_);
  // fmt::print("{}", plan_);
}

Plan NaivePlanGenerator::pipeline_() {
  size_t numH2DChunks = (H2DTraffic_ + bufferSize_ - 1) / bufferSize_;
  size_t numD2HChunks = (D2HTraffic_ + bufferSize_ - 1) / bufferSize_;
  size_t totalIt = (std::max(numH2DChunks, numD2HChunks) + 3) / 4;

  std::vector<std::pair<size_t, size_t>> H2DChunks, D2HChunks;
  for(size_t acc = 0; acc < H2DTraffic_; acc += bufferSize_) {
    H2DChunks.push_back(std::make_pair(acc, std::min(H2DTraffic_, acc + bufferSize_)));
  }
  for(size_t acc = 0; acc < D2HTraffic_; acc += bufferSize_) {
    D2HChunks.push_back(std::make_pair(acc, std::min(D2HTraffic_, acc + bufferSize_)));
  }

  std::vector<size_t> H2DLinksEndIt(4, 0), D2HLinksEndIt(4, 0);
  size_t leftChunks = numH2DChunks, i = 0;
  while (leftChunks) {
    size_t curIt = std::min(leftChunks, totalIt);
    H2DLinksEndIt[i] = curIt;
    leftChunks -= curIt;
    i++;
  }
  leftChunks = numD2HChunks, i = 0;
  while (leftChunks) {
    size_t curIt = std::min(leftChunks, totalIt);
    D2HLinksEndIt[i] = curIt;
    leftChunks -= curIt;
    i++;
  }

  std::vector<ParallelOp> r;

  size_t curH2DChunkId = 0, curD2HChunkId = 0;
  int cur = 1, next = 0;
  for (i = 0; i < totalIt; i++) {
    ParallelOp op;
    for (int d = 0; d < 4; d++) {
      if (i < H2DLinksEndIt[d]) {
        auto &[beg, end] = H2DChunks[curH2DChunkId];
        if (d != deviceId_) { // indirect H2D
          op.emplace_back(MemoryID{d, next, 0,  end - beg}, MemoryID{-1, 0, beg, end});
        } else { // direct H2D
          op.emplace_back(MemoryID{d, 0, beg, end}, MemoryID{-1, 0, beg, end});
        }
        curH2DChunkId++;
      }
    }

    for (int d = 0; d < 4; d++) {
      if (i < D2HLinksEndIt[d]) {
        auto &[beg, end] = D2HChunks[curD2HChunkId];
        if (d != deviceId_) { // indirect D2H
          op.emplace_back(MemoryID{-1, 1, beg, end}, MemoryID{d, 2 + next, 0, end - beg});
        } else { // direct D2H
          op.emplace_back(MemoryID{-1, 1, beg, end}, MemoryID{d, 1, beg, end});
        }
        curD2HChunkId++;
      }
    }
    r.push_back(op);

    std::swap(cur, next);
  }
  assert(curH2DChunkId == numH2DChunks);
  assert(curD2HChunkId == numD2HChunks);

  return r;
}

Plan NaivePlanGenerator::complete_(const Plan &plan) {
  Plan result(plan.size() + 2);

  for (int step = 0; step < plan.size(); step++) {
    auto &parOps = plan[step];

    int i = step + 1;
    for (auto &op: parOps) {
      auto &[opDst, opSrc] = op;
      int opDstID = std::get<0>(opDst), opSrcID = std::get<0>(opSrc);

      if (opSrcID == -1) { // H2D
        if (opDstID != deviceId_) { // indirect
          size_t sendBeg, sendEnd;
          std::tie(std::ignore, std::ignore, sendBeg, sendEnd) = opSrc;
          result[i + 1].emplace_back(MemoryID{deviceId_, 0, sendBeg, sendEnd}, opDst);
        }
      } else { // D2H
        if (opSrcID != deviceId_) { // indirect
          size_t sendBeg, sendEnd;
          std::tie(std::ignore, std::ignore, sendBeg, sendEnd) = opDst;
          result[i - 1].emplace_back(opSrc, MemoryID{deviceId_, 1, sendBeg, sendEnd});
        }
      }
      result[i].push_back(op);
    }
  }
  return result;
}

MemoryRef PlanContext::bindMem(const MemoryID &mem) {
  auto &[dev, id, beg, end] = mem;
  if (dev == -1) {
    if (id == 0) { // H2D
      return srcHost_.slice(beg, end);
    } else { // D2H
      return dstHost_.slice(beg, end);
    }
  } else if (dev == dstDevice_.device) {
    if (id == 0) { // H2D
      return dstDevice_.slice(beg, end);
    } else { // D2H
      return srcDevice_.slice(beg, end);
    }
  } else {
    return MemoryRef{stage_.buffer(dev, id)}.slice(beg, end);
  }
}


BlockPlanExecutor::BlockPlanExecutor(PlanContext &context, const Plan &plan)
  : context_(context), plan_(plan) {}

void BlockPlanExecutor::lanuch() {
  for (int i = 0; i < plan_.size(); i++) {
    std::vector<Stream *> toFasten;
    for (const auto &p: plan_[i]) {
      auto [dstRef, srcRef, stream] = context_.bind(p);
      MemcpyAsync(dstRef, srcRef, *stream);
      toFasten.push_back(stream);
    }
    for (auto &s: toFasten) {
      s->synchronize();
    }
  }
}

void BlockPlanExecutor::lanuch_s() {
  for (int i = 0; i < plan_.size(); i++) {
    // std::vector<Stream *> toFasten;
    for (const auto &p: plan_[i]) {
      auto [dstRef, srcRef, stream] = context_.bindByBuf(p);
      MemcpyAsync(dstRef, srcRef, *stream);
      // toFasten.push_back(stream);
    }
    // for (auto &s: toFasten) {
    //   s->synchronize();
    // }
  }
  context_.stage_.synchronize();
}

void BlockPlanExecutor::lanuch_ss() {
  std::vector<std::vector<gpuio::hip::Event>> h2dE, d2hE;
  std::vector<int> lastOnH2d(4, 0), lastOnD2h(4, 0);

  for (int i = 0; i < plan_.size(); i++) {
    std::vector<gpuio::hip::Event> curH2dE, curD2hE;
    std::vector<int> curOnH2d(4, 0), curOnD2h(4, 0);
    for (int d = 0; d < 4; d++) {
      gpuio::hip::DeviceGuard on(d);
      curH2dE.emplace_back();
      curD2hE.emplace_back();
    }

    for (const auto &p: plan_[i]) {
      auto [dstRef, srcRef, stream] = context_.bindByBuf(p);
      // assume the device is 0
      bool isI2H = dstRef.device == -1 && srcRef.device != 0;
      bool isH2I = srcRef.device == -1 && dstRef.device != 0;
      if (isI2H) { 
        if (lastOnD2h[srcRef.device]) {
          stream->waitEvent(d2hE.back()[srcRef.device]);
        }
      }
      if (isH2I) {
        if (lastOnH2d[dstRef.device]) {
          stream->waitEvent(h2dE.back()[dstRef.device]); 
        }
      }
      MemcpyAsync(dstRef, srcRef, *stream);
      if (isI2H) { 
        curD2hE[srcRef.device].record(*stream); 
        curOnD2h[srcRef.device] = 1;
      }
      if (isH2I) { 
        curH2dE[dstRef.device].record(*stream); 
        curOnH2d[dstRef.device] = 1;
      }
    }

    h2dE.push_back(std::move(curH2dE));
    d2hE.push_back(std::move(curD2hE));
    lastOnH2d = curOnH2d;
    lastOnD2h = curOnD2h;
  }
  context_.stage_.synchronize();

}

// a strawman design
void BlockPlanExecutor::lanuch_gg() {
  // The most naive scheduling method
  std::vector<std::vector<gpuio::hip::Event>> h2dE, d2hE;
  std::vector<int> lastOnH2d(4, 0), lastOnD2h(4, 0);

  for (int i = 0; i < plan_.size(); i++) {
    std::vector<gpuio::hip::Event> curH2dE, curD2hE;
    std::vector<int> curOnH2d(4, 0), curOnD2h(4, 0);
    for (int d = 0; d < 4; d++) {
      gpuio::hip::DeviceGuard on(d);
      curH2dE.emplace_back();
      curD2hE.emplace_back();
    }

    for (const auto &p: plan_[i]) {
      auto [dstRef, srcRef, stream] = context_.bindByBuf(p);
      // first schedule all the H2D transfer
      if (srcRef.device == 0 || dstRef.device == -1) continue;
      // assume the device is 0
      bool isI2H = dstRef.device == -1 && srcRef.device != 0;
      bool isH2I = srcRef.device == -1 && dstRef.device != 0;
      if (isI2H) { 
        if (lastOnD2h[srcRef.device]) {
          stream->waitEvent(d2hE.back()[srcRef.device]);
        }
      }
      if (isH2I) {
        if (lastOnH2d[dstRef.device]) {
          stream->waitEvent(h2dE.back()[dstRef.device]); 
        }
      }
      MemcpyAsync(dstRef, srcRef, *stream);
      if (isI2H) { 
        curD2hE[srcRef.device].record(*stream); 
        curOnD2h[srcRef.device] = 1;
      }
      if (isH2I) { 
        curH2dE[dstRef.device].record(*stream); 
        curOnH2d[dstRef.device] = 1;
      }
    }

    h2dE.push_back(std::move(curH2dE));
    d2hE.push_back(std::move(curD2hE));
    lastOnH2d = curOnH2d;
    lastOnD2h = curOnD2h;
  }

  std::vector<std::vector<gpuio::hip::Event>> h2dE_2, d2hE_2;
  std::vector<int> lastOnH2d_2(4, 0), lastOnD2h_2(4, 0);

  for (int i = 0; i < plan_.size(); i++) {
    std::vector<gpuio::hip::Event> curH2dE, curD2hE;
    std::vector<int> curOnH2d(4, 0), curOnD2h(4, 0);
    for (int d = 0; d < 4; d++) {
      gpuio::hip::DeviceGuard on(d);
      curH2dE.emplace_back();
      curD2hE.emplace_back();
    }

    for (const auto &p: plan_[i]) {
      auto [dstRef, srcRef, stream] = context_.bindByBuf(p);
      // first schedule all the H2D transfer
      if (srcRef.device == -1 || dstRef.device == 0) continue;
      // assume the device is 0
      bool isI2H = dstRef.device == -1 && srcRef.device != 0;
      bool isH2I = srcRef.device == -1 && dstRef.device != 0;
      if (isI2H) { 
        if (lastOnD2h_2[srcRef.device]) {
          stream->waitEvent(d2hE_2.back()[srcRef.device]);
        }
      }
      if (isH2I) {
        if (lastOnH2d_2[dstRef.device]) {
          stream->waitEvent(h2dE_2.back()[dstRef.device]); 
        }
      }
      MemcpyAsync(dstRef, srcRef, *stream);
      if (isI2H) { 
        curD2hE[srcRef.device].record(*stream); 
        curOnD2h[srcRef.device] = 1;
      }
      if (isH2I) { 
        curH2dE[dstRef.device].record(*stream); 
        curOnH2d[dstRef.device] = 1;
      }
    }

    h2dE_2.push_back(std::move(curH2dE));
    d2hE_2.push_back(std::move(curD2hE));
    lastOnH2d_2 = curOnH2d;
    lastOnD2h_2 = curOnD2h;
  }

  context_.stage_.synchronize();

}

} // namespace gpuio::sched::profile

namespace gpuio::sched::dyn {

void sleep_for(std::chrono::microseconds v) {
  using namespace std::chrono;
  auto n = high_resolution_clock::now();
  while (n + v > high_resolution_clock::now()) continue;
}

void LoadBalancingSched::_produceTasks(std::vector<IOTask> &tasks, MemoryRef dst, MemoryRef src, size_t gran) {
  assert(dst.size == src.size);

  for (size_t i = 0; i < dst.size; i += gran) {
    size_t beg = i, end = std::min(dst.size, i + gran);
    int id = tasks.size();
    tasks.emplace_back(IOTask{dst.slice(beg, end), src.slice(beg, end), id, false});
  }
}

void LoadBalancingSched::reset(
  const std::vector<MemoryRef> &H2Ddsts, const std::vector<MemoryRef> &H2Dsrcs,
  const std::vector<MemoryRef> &D2Hdsts, const std::vector<MemoryRef> &D2Hsrcs,
  size_t gran
) {
  h2dCnt_ = 0;
  d2hCnt_ = 0;
  h2dIssued_ = 0;
  d2hIssued_ = 0;
  H2Dtraffic_ = 0;
  D2Htraffic_ = 0;

  h2dTasks_.clear();
  d2hTasks_.clear();
  for (size_t i = 0; i < H2Ddsts.size(); i++) {
    H2Dtraffic_ += H2Ddsts[i].size;
    _produceTasks(h2dTasks_, H2Ddsts[i], H2Dsrcs[i], gran);
  }
  for (size_t i = 0; i < D2Hdsts.size(); i++) {
    D2Htraffic_ += D2Hdsts[i].size;
    _produceTasks(d2hTasks_, D2Hdsts[i], D2Hsrcs[i], gran);
  }

  std::fill(robh2d_.begin(), robh2d_.end(), 0);
  std::fill(robd2h_.begin(), robd2h_.end(), 0);
  assert(h2dTasks_.size() < 10000);
  assert(d2hTasks_.size() < 10000);

  double readRatio = (H2Dtraffic_ * 1.0) / (H2Dtraffic_ + D2Htraffic_);
  if (readRatio < 0.75) {
    fixedD2H = 1;
  }
}

void LoadBalancingSched::reset(MemoryRef H2Ddst, MemoryRef H2Dsrc, MemoryRef D2Hdst, MemoryRef D2Hsrc, size_t gran) {
  reset(
    std::vector<MemoryRef>{H2Ddst}, std::vector<MemoryRef>{H2Dsrc},
    std::vector<MemoryRef>{D2Hdst}, std::vector<MemoryRef>{D2Hsrc},
    gran
  );
}

IOTask LoadBalancingSched::nextH2D(int id) {
  size_t i = h2dCnt_.fetch_add(1);
  if (i < h2dTasks_.size()) {
    h2dIssued_.fetch_add(h2dTasks_[i].dst.size);
    return h2dTasks_[i];
  } else {
    return IOTask{MemoryRef{}, MemoryRef{}, -1, true};
  }
}

IOTask LoadBalancingSched::nextD2H(int id) {
  if (id > 4 + fixedD2H) { // hyper paramter tuning point
    size_t H2D_equiv = (1.0 * h2dIssued_) / H2Dtraffic_ * D2Htraffic_;
    size_t cur_d2h = d2hIssued_;
    bool pause = cur_d2h > H2D_equiv;
    if (pause) {
      return IOTask{MemoryRef{}, MemoryRef{}, -1, false};
    }
  }

  size_t i = d2hCnt_.fetch_add(1);
  if (i < d2hTasks_.size()) {
    d2hIssued_.fetch_add(d2hTasks_[i].dst.size);
    return d2hTasks_[i];
  } else {
    return IOTask{MemoryRef{}, MemoryRef{}, -1, true};
  }
}

IOTask LoadBalancingSched::next(int id) {
  if (id < 4) {
    return nextH2D(id);
  } else {
    return nextD2H(id);
  }
}

void LoadBalancingSched::report(int id, int task) {
  if (id < 4) {
    robh2d_[task] = 1;
  } else {
    robd2h_[task] = 1;
  }
}

void LoadBalancingSched::h2dSent() {
  for (size_t i = 0; i < h2dTasks_.size(); i++) {
    while (robh2d_[i] != 1) continue;
    size_t end = std::min((i + 1) * gran_, H2Dtraffic_);
    h2dcallback_(end);
  }
}

void LoadBalancingSched::d2hSent() {
  for (size_t i = 0; i < d2hTasks_.size(); i++) {
    while (robd2h_[i] != 1) continue;
    size_t end = std::min((i + 1) * gran_, D2Htraffic_);
    d2hcallback_(end);
  }
}

void IndirectLink::caller() {
  int old1{0}, old2{0};
  while (true) {
    // fmt::print("[{}] loop begin\n", id_);
    size_t curBufSize = bufDst_.size;
    int curSentId = bufTaskId_;
    if (curBufSize > 0) {
      gpuio::hip::MemcpyAsync(bufDst_, bufs_[cur_].slice(0, curBufSize), *streams_[1]); // empty current buffer
      bufDst_ = MemoryRef{};
      bufTaskId_ = -1;
    }

    auto task = nextFn_(id_);
    // fmt::print("[{}] got task\n", id_);
    size_t nextBufSize = task.src.size;
    if (nextBufSize > 0) {
      gpuio::hip::MemcpyAsync(bufs_[next_].slice(0, nextBufSize), task.src, *streams_[0]);
      bufDst_ = task.dst;
      bufTaskId_ = task.id;
    }
    // fmt::print("[{}] sent s1\n", id_);

    if (curBufSize > 0) {
      gpuio::hip::addHostCallback<IndirectLink, &IndirectLink::s2Callback>(*streams_[1], this);
    }
    if (nextBufSize > 0) {
      gpuio::hip::addHostCallback<IndirectLink, &IndirectLink::s1Callback>(*streams_[0], this);
    }
    // fmt::print("[{}] added callback, lastSentId_: {} \n", id_, lastSentId_);

    if (lastSentId_ >= 0) { // report back the finished data
      reportFn_(id_, lastSentId_);
      lastSentId_ = -1;
    }
    lastSentId_ = curSentId;

    // swap buffer pointer
    std::swap(cur_, next_);

    if (curBufSize > 0) {
      while (old2 == stage2Cnt_) continue; // wait until previous transfer all finish
    }
    if (nextBufSize > 0) {
      while (old1 == stage1Cnt_) continue;
    }
    if (!task.done && nextBufSize == 0 && curBufSize == 0) { // scheduler pause the link
      using namespace std::chrono_literals;
      sleep_for(10us);
    }
    old1 = stage1Cnt_;
    old2 = stage2Cnt_;

    if (task.done) break; // last transfer, break
  }

  if (lastSentId_ >= 0) { // report back the finished data
    reportFn_(id_, lastSentId_);
    lastSentId_ = -1;
  }

}

void DirectLink::caller() {
  int old{0};
  while (true) {
    auto task = nextFn_(id_);
    if (task.dst.size > 0) {
      gpuio::hip::MemcpyAsync(task.dst, task.src, *s_); // empty current buffer
      gpuio::hip::addHostCallback<DirectLink, &DirectLink::callback>(*s_, this);
      // fmt::print("[{}] sent s1\n", id_);
      while (old == cnt_) continue;
      if (task.dst.size > 0) {
        reportFn_(id_, task.id);
      }
      old = cnt_;
    }

    if (task.done) break;
  }
}


ExchangeContextOwning::ExchangeContextOwning(size_t gran): gran_(gran) {
  for (int d = 0; d < 4; d++) {
    gpuio::hip::DeviceGuard on(d);
    h2dbufs_.emplace_back(gran_);
    h2dbufs_.emplace_back(gran_);
    d2hbufs_.emplace_back(gran_);
    d2hbufs_.emplace_back(gran_);
    h2dstreams_.emplace_back();
    h2dstreams_.emplace_back();
    d2hstreams_.emplace_back();
    d2hstreams_.emplace_back();
  }
}

void ExchangeContextOwning::warmup(MemoryRef host) {
  for (int d = 0; d < 4; d++) {
    for (int s = 0; s < 2; s++) {
      auto &h2ds = h2dS(d, s);
      auto h2db = h2dbufs(d, s);
      gpuio::hip::MemcpyAsync(host, h2db, h2ds);
      for (int dj = 0; dj < 4; dj++) {
        if (dj != d) {
          auto nb = h2dbufs(dj, s);
          gpuio::hip::MemcpyAsync(nb, h2db, h2ds);
        }
      }
      gpuio::hip::addHostCallback<ExchangeContextOwning, &ExchangeContextOwning::_warmupfunc>(h2ds, this);
      h2ds.synchronize();

      auto &d2hs = d2hS(d, s);
      auto d2hb = d2hbufs(d, s);
      gpuio::hip::MemcpyAsync(host, d2hb, d2hs);
      for (int dj = 0; dj < 4; dj++) {
        if (dj != d) {
          auto nb = d2hbufs(dj, s);
          gpuio::hip::MemcpyAsync(nb, d2hb, d2hs);
        }
      }
      gpuio::hip::addHostCallback<ExchangeContextOwning, &ExchangeContextOwning::_warmupfunc>(d2hs, this);
      d2hs.synchronize();
    }
  }
}



} // namespace gpuio::sched::dyn
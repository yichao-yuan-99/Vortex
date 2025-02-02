#include <fmt/core.h>
#include <hipWrapper.h>

int main() {
  fmt::print("Number of device: {}\n", gpuio::hip::platform.deviceCount());
  fmt::print("{:-<34}\n", "");
  fmt::print("Peer to Peer Access:\n");
  gpuio::misc::printPeerAccessStatus();
  for (int i = 0; i < gpuio::hip::platform.deviceCount(); i++) {
    gpuio::misc::printDeviceProp(gpuio::hip::platform.deviceProp(i));
  }
}
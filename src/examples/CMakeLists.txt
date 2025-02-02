project(GPU-IO-examples LANGUAGES CXX)

add_executable(hostCallback hostCallback.cc)
target_link_libraries(hostCallback gpuio::hip)
# target_link_libraries(hostCallback fmt::fmt)

add_executable(deviceProperties deviceProperties.cc)
target_link_libraries(deviceProperties gpuio::hip)

add_executable(bandwidthProfile bandwidthProfile.cc)
target_link_libraries(bandwidthProfile gpuio::sched)

add_executable(ioplan ioplan.cc)
target_link_libraries(ioplan gpuio::sched)
target_link_libraries(ioplan argparse)

add_executable(dynio dynio.cc)
target_link_libraries(dynio gpuio::sched)
target_link_libraries(dynio argparse)

add_executable(hashtable hashtable.cc)
target_link_libraries(hashtable gpuio::hip)
target_link_libraries(hashtable gpuio::kernels)
# target_link_libraries(hashtable range-v3)

add_executable(gpubandwidth gpubandwidth.cc)
target_link_libraries(gpubandwidth gpuio::hip)

add_executable(sort sort.cc)
target_link_libraries(sort gpuio::hip)
target_link_libraries(sort gpuio::sched)
target_link_libraries(sort gpuio::kernels)
target_link_libraries(sort gpuio::execution)
target_link_libraries(sort tbb)

add_executable(sort-alt sort-alt.cc)
target_link_libraries(sort-alt gpuio::hip)
target_link_libraries(sort-alt gpuio::sched)
target_link_libraries(sort-alt gpuio::kernels)
target_link_libraries(sort-alt gpuio::execution)
target_link_libraries(sort-alt tbb)

add_executable(join join.cc)
target_link_libraries(join gpuio::hip)
target_link_libraries(join gpuio::sched)
target_link_libraries(join gpuio::kernels)
target_link_libraries(join gpuio::execution)
target_link_libraries(join tbb)

add_executable(gpuAsDMA gpuAsDMA.cc)
target_link_libraries(gpuAsDMA gpuio::hip)

add_executable(zero-copy zero-copy.cc)
target_link_libraries(zero-copy gpuio::hip)
target_link_libraries(zero-copy gpuio::sched)
target_link_libraries(zero-copy gpuio::kernels)
target_link_libraries(zero-copy gpuio::execution)
target_link_libraries(zero-copy tbb)

add_executable(sumSelective sumSelective.cc)
target_link_libraries(sumSelective gpuio::hip)
target_link_libraries(sumSelective gpuio::sched)
target_link_libraries(sumSelective gpuio::kernels)
target_link_libraries(sumSelective gpuio::execution)
target_link_libraries(sumSelective tbb)
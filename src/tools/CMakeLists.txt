project(GPU-IO-TOOLS LANGUAGES CXX)

add_executable(permutation permutation.cc)
target_link_libraries(permutation argparse)
target_link_libraries(permutation fmt::fmt)

add_executable(uniqueKeys uniqueKeys.cc)
target_link_libraries(uniqueKeys argparse)
target_link_libraries(uniqueKeys fmt::fmt)
target_link_libraries(uniqueKeys gpuio::hip)
target_link_libraries(uniqueKeys tbb)

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

execute_process(COMMAND hipconfig --rocmpath OUTPUT_VARIABLE rocm_path)
message(STATUS "ROCm SDK path: ${rocm_path}")
set(CMAKE_CXX_COMPILER "${rocm_path}/bin/hipcc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${rocm_path}/include")

project(GPU-IO LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(hip)

# CPM
set(CPM_DOWNLOAD_LOCATION ${CMAKE_BINARY_DIR}/CPM.cmake)
file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/cpm.cmake ${CPM_DOWNLOAD_LOCATION})
include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage("gh:fmtlib/fmt#10.2.1")
CPMAddPackage("gh:p-ranav/argparse#v3.0")
CPMAddPackage("gh:oneapi-src/oneTBB#v2021.12.0")
# change this
find_package(rocprim)
# find_package(rocprim REQUIRED CONFIG PATHS "/home1/yichaoy/work/opt/rocPRIM")
# find_package(rocthrust REQUIRED CONFIG PATHS "/home1/yichaoy/work/opt/rocThrust")
# CPMAddPackage("gh:ROCm/rocPRIM#rocm-6.0.2")
# CPMAddPackage("gh:ROCm/rocThrust#rocm-6.0.2")
# cannot use range to accumulate filtered numbers, do not know why
# CPMAddPackage("gh:ericniebler/range-v3#0.12.0")

# set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src)
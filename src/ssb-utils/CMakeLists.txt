project(GPU-IO-SSB-UTILS LANGUAGES CXX)

add_library(${PROJECT_NAME} src/ssb-utils.cc)
add_library(gpuio::ssb-utils ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PUBLIC gpuio::hip)
target_link_libraries(${PROJECT_NAME} PUBLIC gpuio::execution)
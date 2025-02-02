project(GPU-IO-KERNELS LANGUAGES CXX)

add_library(${PROJECT_NAME} src/sort.cc)
add_library(gpuio::kernels ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE roc::rocprim)
target_link_libraries(${PROJECT_NAME} PUBLIC gpuio::hip)
# target_link_libraries(${PROJECT_NAME} PRIVATE hip::device)
project(GPU-IO-EXECUTION LANGUAGES CXX)

add_library(${PROJECT_NAME} INTERFACE)
add_library(gpuio::execution ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} INTERFACE gpuio::kernels)
target_link_libraries(${PROJECT_NAME} INTERFACE gpuio::sched)
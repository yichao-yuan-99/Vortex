project(GPU-IO-SCHED LANGUAGES CXX)


add_library(${PROJECT_NAME} src/io-sched.cc)
add_library(gpuio::sched ALIAS ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PUBLIC gpuio::hip)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
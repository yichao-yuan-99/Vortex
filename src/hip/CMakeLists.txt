project(GPU-IO-HIP LANGUAGES CXX)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(${PROJECT_NAME} src/hipWrapper.cc)
# add_library(${PROJECT_NAME} src/hipWarpper.hip)
add_library(gpuio::hip ALIAS ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PUBLIC hip::host)
target_link_libraries(${PROJECT_NAME} PUBLIC roc::rocrand)
target_link_libraries(${PROJECT_NAME} PUBLIC fmt::fmt)


target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
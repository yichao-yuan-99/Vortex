project(GPU-IO-crystal-ssb-zero-copy LANGUAGES CXX)

macro(SSBQUERY Q)

add_executable(${Q}-s ${Q}.cc)
target_link_libraries(${Q}-s gpuio::sched)
target_link_libraries(${Q}-s gpuio::execution)
target_link_libraries(${Q}-s gpuio::ssb-utils)
target_link_libraries(${Q}-s gpuio::kernels)
  
endmacro()


SSBQUERY(q11)
SSBQUERY(q12)
SSBQUERY(q13)
SSBQUERY(q21)
SSBQUERY(q22)
SSBQUERY(q23)
SSBQUERY(q31)
SSBQUERY(q32)
SSBQUERY(q33)
SSBQUERY(q34)
SSBQUERY(q41)
SSBQUERY(q42)
SSBQUERY(q43)
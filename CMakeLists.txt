cmake_minimum_required(VERSION 3.15)

project(graph_cuts CUDA CXX)

include(FetchContent)

FetchContent_Declare(
  GoogleBenchmark
  URL https://github.com/google/benchmark/archive/v1.4.1.tar.gz
  )

if (NOT GoogleBenchmark_POPULATED)
  FetchContent_Populate(GoogleBenchmark)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
  add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()



find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

add_executable(graph_cuts
        cpu_implem/src/main.cc
        cpu_implem/src/node.cc
        cpu_implem/src/graph.cc
        cpu_implem/src/gpu.cu
        )

add_executable(bench
        cpu_implem/src/bench.cc
        cpu_implem/src/graph.cc
        cpu_implem/src/gpu.cu
        )
target_link_libraries(bench PRIVATE benchmark)

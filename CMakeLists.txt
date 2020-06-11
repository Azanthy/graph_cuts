cmake_minimum_required(VERSION 3.15)

project(graph_cuts CUDA CXX)

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

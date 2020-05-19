cmake_minimum_required(VERSION 3.15)

project(graph_cuts)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

add_executable(graph_cuts
        cpu_implem/src/main.cc
        )

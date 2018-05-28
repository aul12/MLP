cmake_minimum_required(VERSION 3.10)
project(MlpTest)

set(CMAKE_CXX_STANDARD 17)

add_executable(MlpTest main.cpp Layer.hpp Mlp.hpp)
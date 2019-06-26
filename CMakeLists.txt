cmake_minimum_required(VERSION 3.10)

# Build options
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Werror -march=native -mtune=native")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
else()
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3")
endif ()

# Building
project(Mlp)

include_directories(lib)

add_executable(MlpTest main.cpp)
target_link_libraries(MlpTest)

add_library(${PROJECT_NAME} INTERFACE)

# Installation via make install
include(GNUInstallDirs)
install(FILES Layer.hpp Mlp.hpp Functions.hpp lib/json.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Mlp)

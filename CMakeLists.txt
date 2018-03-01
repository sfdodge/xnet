# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# c++ version
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define project name
project(xNet)

# xtensor
find_package(xtensor REQUIRED)
include_directories(${XTENSOR_INCLUDE_DIRS})
find_package(xtensor-blas REQUIRED)
include_directories(${XTENSOR_BLAS_INCLUDE_DIRS})

# catch for testing
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE "lib")

# Declare the executable target built from your sources
file(GLOB main_src
  "src/*.cpp"
  "src/*.hpp"
  )
set(mnist_src "examples/mnist.cpp")
file(GLOB test_src
  "test/*.cpp"
  )
include_directories("lib/")

# add_executable(xNet ${main_src})
add_executable(mnist_example ${main_src} ${mnist_src})
add_executable(test_xNet ${main_src} ${test_src})

target_link_libraries(test_xNet Catch)

# Debug build
# set(CMAKE_BUILD_TYPE Debug)

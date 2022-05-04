cmake_minimum_required(VERSION 3.7)
project(NN-MPC)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O2")

include_directories(.)

find_package(Python COMPONENTS Development)
find_package(Torch REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main ${Python_LIBRARIES})
target_link_libraries(main ${TORCH_LIBRARIES})

set_property(TARGET main PROPERTY CXX_STANDARD 14)

cmake_minimum_required(VERSION 3.7)
project(NN-MPC)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O2")

include_directories(.)
include_directories(External/blasfeo/lib/include)
include_directories(External/hpipm/lib/include)
include_directories(External/matplotlib)
include_directories(External/Eigen)
include_directories(External/Json/include)

add_executable(NN-MPC
        main.cpp
        torch.cpp
        types.cpp
        types.h
        config.h
        Params/params.cpp
        Spline/cubic_spline.cpp
        Spline/arc_length_spline.cpp
        Interfaces/hpipm_interface.cpp
        Interfaces/solver_interface.cpp
        Constraints/constraints.cpp
        Constraints/bounds.cpp
        Cost/cost.cpp
        Model/model.cpp
        Model/integrator.cpp
        Tests/spline_test.cpp
        Tests/cost_test.cpp
        Tests/model_integrator_test.cpp
        Tests/constratins_test.cpp
        MPC/mpc.cpp
        Params/track.cpp
        Params/track.h
        Plotting/plotting.cpp
        Plotting/plotting.h)

find_package(Python COMPONENTS Development)
find_package(Torch REQUIRED)

target_include_directories(NN-MPC PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(NN-MPC ${Python_LIBRARIES})
target_link_libraries(NN-MPC ${TORCH_LIBRARIES})

target_link_libraries(NN-MPC ${CMAKE_SOURCE_DIR}/External/hpipm/lib/lib/libhpipm.a ${CMAKE_SOURCE_DIR}/External/blasfeo/lib/lib/libblasfeo.a m)

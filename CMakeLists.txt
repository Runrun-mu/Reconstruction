cmake_minimum_required(VERSION 3.10)
project(reconstruction)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS math_c99 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(PythonLibs 3.8 REQUIRED)

add_executable(reconstruction predict.hpp unwrap.hpp main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(../plot/matplotlib-cpp)
target_link_libraries(reconstruction Eigen3::Eigen ${OpenCV_LIBS} Boost::math_c99 stdc++fs ${PYTHON_LIBRARIES})

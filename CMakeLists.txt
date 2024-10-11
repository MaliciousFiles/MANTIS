cmake_minimum_required(VERSION 3.29)
project(MANTIS)

# **** VARIABLES ****
set(EIGEN_INCLUDE /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN_INCLUDE})

add_executable(MANTIS src/main.cpp
        src/util/gate.h
        src/util/gate.cpp
        src/util/circbuf.h
)
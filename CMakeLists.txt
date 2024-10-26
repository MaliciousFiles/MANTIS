cmake_minimum_required(VERSION 3.29)
project(MANTIS)

# **** VARIABLES ****
set(EIGEN_INCLUDE /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3)
set(MATPLOT_INCLUDE /opt/homebrew/Cellar/matplotplusplus/1.2.1/include)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN_INCLUDE})

find_package(Matplot++ REQUIRED)
include_directories(${MATPLOT_INCLUDE})

add_executable(MANTIS src/main.cpp
        src/lstm/gate.h
        src/lstm/gate.cpp
        src/util/circbuf.h
        src/lstm/lstm.h
        src/lstm/lstm.cpp
        src/util/math.h
        src/util/math.cpp
)

target_link_libraries(MANTIS Matplot++::matplot)
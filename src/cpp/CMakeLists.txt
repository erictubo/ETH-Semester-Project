cmake_minimum_required(VERSION 3.5)

project(optimization)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_INSTALL_PREFIX .)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
find_package(gflags REQUIRED)
# find_package(SuiteSparse REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})

find_package(pybind11 REQUIRED)
# add_subdirectory(pybind11)

pybind11_add_module(optimization optimization.cc)

add_executable(optimization-exec optimization.cc)

target_link_libraries(optimization 
  PRIVATE
    ${CERES_LIBRARIES}
    ${MY_LIBRARIES}
    pybind11::embed)

target_include_directories(optimization
  PRIVATE
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIRS}
    # ${SUITESPARSE_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

install (TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ..
)
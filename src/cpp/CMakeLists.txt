cmake_minimum_required(VERSION 3.12)

project(optimization)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_INSTALL_PREFIX .)

# Set the policy for FindPython
# if(POLICY CMP0148)
#     cmake_policy(SET CMP0148 NEW)
# endif()

# Find Python and its libraries
# find_package(Python 3.9.6 EXACT COMPONENTS Interpreter Development REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(gflags REQUIRED)
# find_package(glog REQUIRED)
# find_package(SuiteSparse REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CERES_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# find_package(pybind11 REQUIRED)
add_subdirectory(pybind11)

pybind11_add_module(optimization optimization.cc)

# add_executable(optimization-exec optimization.cc)

target_link_libraries(optimization 
  PRIVATE
    ${CERES_LIBRARIES}
    ${OpenCV_LIBS}
    # ${MY_LIBRARIES}
    # pybind11::embed
    pybind11::pybind11
    )

# Install the target
install (TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ..
)
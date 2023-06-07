# Multi-Frame Extrinsics Estimation
**<u>Project Title: Online Extrinsic Camera Calibration from Multiple Keyframes using Map Information</u>**

This project implements an algorithm to compute the relative pose between a GPS sensor and a monocular, intrinsically-calibrated camera at the front of a track vehicle (mounted rigidly in an unknown location that is to be inferred from its images).

Building upon the previous work done by Nicolina (branch: main) this project aims to come up with a more accurate and robust estimation process. To achieve this, map information is combined with detected railway tracks and an iterative approach is used that leverages information across multiple keyframes.

Note that the detection pipeline is not implemented (yet), but annotated features are used to simulate their input.

**Axis Definitions & Coordinate Systems**

| Direction                 | Rotation | GPS axis   | Camera axis |
|---------------------------|----------|------------|-------------|
| Longitudinal (forward)    | Roll     | $+X_{GPS}$ | $+Z_C$      |
| Lateral (sideways, right) | Pitch    | $+Y_{GPS}$ | $+X_C$      |
| Vertical (upwards)        | Yaw      | $+Z_{GPS}$ | $-Y_C$      |


**Python Packages**

- numpy
- matplotlib
- cv2
- pickle
- math
- random
- ...


**Compilation of C++ & Pybind (Ceres Solver)**

Compile in build directory using CMake

```console
cd src/cpp/
mkdir build
cd build
cmake ..
```


Make executable, include all dependencies (-I) and link to Ceres (-lceres)


```console
cd -

g++ -std=c++17 -I/usr/local/include/eigen3/ -I/usr/local/include/glog/ -I/usr/local/include/gflags/ -I/usr/local/include/pybind11/ -I/usr/local/include/python3.9 -o optimization -undefined dynamic_lookup $(python3-config --includes) optimization.cc -o optimization$(python3-config --extension-suffix) -lceres
````

Linux: replace "-undefined dynamic_lookup" with "-fPIC"

More info / troubleshooting: https://pybind11.readthedocs.io/en/latest/compiling.html#building-manually

This process reates a CPython file, from which Python can access functions.

**Contact**

Author: Eric Tüschenbönner

E-mail: etueschenboe@student.ethz.ch



**Linux Build**

Compile the C++ optimization file.

```console
cd src/cpp
mkdir build
cd build
cmake ..
make
make install

cd src

python3 interface.py
```


delete build directory
delete "optimization.cpython---.so" file
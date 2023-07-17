# Multi-Frame Extrinsics Optimization

**<b>"Online Extrinsic Camera Calibration from Multiple Keyframes using Map Information"</b>**

This project implements an algorithm to compute the relative pose between a GPS sensor and an intrinsically-calibrated camera at the front of a track vehicle (mounted rigidly in an unknown location that is to be inferred from its images). Building upon the previous work by Nicolina (branch: main) this project aims to come up with a more reliable and generalizable estimation process. To achieve this, map information is combined with detected railway tracks using an optimization approach based on iterative closest points (ICP) that leverages information across multiple frames. Note that the detection pipeline is not implemented yet; instead annotations are used to simulate observed tracks. Moreover, in contrast to single-frame, multi-frame optimization is somewhat limited by data accuracy. To address this, sensor fusion should be implemented combining the GPS data with IMU and odometry information in order to obtain a more precise state estimate.

## Input Data

Store the relevant input data locally and specific the paths in `data.py`. The current structure uses 'path_to_data' as the root directory, which contains the subdirectories 'map', 'elevation', and 'frames'. The latter contains the subdirectories 'frames/poses', 'frames/images' and 'frames/annotations' (the last two for each camera separately).

### Elevation

Obtained automatically when running the pipeline, which calls the method `MapInfo.get_elevation(x_gps, y_gps)`. The data is downloaded from the website https://data.geobasis-bb.de/geobasis/daten/dgm/xyz/ and stored as local files under the specified path `path_to_elevation_data` (e.g. the 'elevation' subdirectory).

### Railway Map (OSM)

Store the relevant OSM file locally (e.g. in the 'map' subdirectory) and specify `path_to_osm_file`. 

### Frames

Each frame contains synchronous data from a stereo camera setup and various sensors, where the same frame corresponds to the same filename (i.e. number 000000).:
- Images for each camera (JPG files)
- Poses from RTK-GPS (YAML files), by extension also from IMU and odometry

This data can be extracted from ROS bags, which are recorded and synchronized using the ROS node `image_gps_sync`. Edit the file `image_gps_sync.py` to export the required data (stereo images and GPS poses, plus optionally IMU and/or odometry) and specify the correct output paths. An edited copy of this file is added to the Git root directory for convenience.

After building the ROS system with all other required files, here are some useful commands to speed up the process:

```terminal
# Initialise ROS
roscore

# Initialise workspace (for each terminal)
catkin_make
source devel/setup.bash

# Run the node
rosrun <package_name> <node_name>

# Play the bag files simultaneously (at a slower/faster rate)
rosbag play <bag_file_1> <bag_file_2> --rate <rate>

# See the available topics from the bag files
rostopic list
```

### Annotations

Manually created using the tool https://www.robots.ox.ac.uk/~vgg/software/via/, by uploading the relevant images and drawing each railway track as a sequence of points. The annotations can be exported as a CSV file, which are read by the `Annotation` class. Finally, specify `path_to_annotations` in `data.py`.

## Python Setup

### Python libraries

See requirements.txt or use directly for quick installation:

```console
pip install -r requirements.txt
```

### File Structure

Interaction files:
- data.py: Specify data locations in file system
- main.py: Main file to run the entire pipeline

Data objects:
- camera.py: Camera class for attributes and specific methods
- railway.py: Railway class with processed OSM & elevation data
    - import_osm.py: Imports OSM data behind the scenes
- keyframe.py: basic Frame & more sophisticated Keyframe class
    - annotation.py: Annotation class, part of Keyframe
    - gps.py: GPS class, part of Keyframe

Static Methods:
- transformation.py: Transformation class
- visualization.py: Visualization class
- map_info.py: MapInfo class for map information retrieval

## C++ Setup

### Pybind11 as Git submodule
Call the following to pull in the pybind11 submodule.
```
git submodule update --init --recursive
```

### Dependencies
- Eigen3 (http://eigen.tuxfamily.org/index.php?title=Main_Page)
- Ceres Solver (http://ceres-solver.org/)
- glog
- gflags
- pybind11
- Python3.9

### C++ Compilation
```console
cd src/cpp/
mkdir build
cd build
cmake ..
```


Make executable, include all dependencies (-I) in their directories and link (-l) the libraries.

The below command is for macOS. For Linux replace "-undefined dynamic_lookup" with "-fPIC". The command assumes the Eigen3, glog, gflags, pybind11 and Python3.9 header files are located in the /usr/local/include/ directory. If this is not the case, specify the correct paths with the -I flag.


```console
cd src/cpp

g++ -std=c++17 -I/usr/local/include/eigen3/ -I/usr/local/include/glog/ -I/usr/local/include/gflags/ -I/usr/local/include/pybind11/ -I/usr/local/include/python3.9 -o optimization -undefined dynamic_lookup $(python3-config --includes) optimization.cc -o optimization$(python3-config --extension-suffix) -lceres
```

More info and/or troubleshooting: https://pybind11.readthedocs.io/en/latest/compiling.html#building-manually

### Pybind11 Compilation

This process creates a CPython file, from which Python can access functions. The file will be created in the same directory as the C++ file with the name `optimization.cpython.<cpython_version>.so`


```console
cd src/cpp/build
make
make install
```
Now, when running the Python file (main.py), it should be able to access the C++ functions.




## Contact

Author: Eric Tüschenbönner

E-mail: etueschenboe@student.ethz.ch
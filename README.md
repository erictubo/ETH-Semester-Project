# Multi-Frame Extrinsics Optimization

[Report](Report.pdf) | [Presentation](Presentation.pdf)

**"Online Extrinsic Camera Calibration from Multiple Keyframes using Map Information"**

This project implements an algorithm to compute the relative pose between a GPS sensor and an intrinsically-calibrated camera at the front of a track vehicle (mounted rigidly in an unknown location that is to be inferred from its images). To achieve this, map information is combined with detected railway tracks using an optimization approach based on iterative closest points (ICP) that leverages information across multiple frames.

Note that the detection pipeline is not implemented; instead annotations are used to simulate observed tracks.

Moreover, in contrast to single-frame, multi-frame optimization has been somewhat limited by data accuracy. To address this, sensor fusion has been implemented via an EKF to combine GPS with IMU data in order to obtain a more precise state estimate.

## Input Data

Store the relevant input data locally and specific the paths in [data.py](src/data.py). The current structure uses `path_to_data` as the root directory, which contains the subdirectories `map`, `elevation`, and `frames`. The latter contains the subdirectories `frames/poses`, `frames/images`, and `frames/annotations` (the last two for each camera separately).

### Elevation

Elevation data is obtained automatically when running the pipeline, which calls the method `MapInfo.get_elevation(x_gps, y_gps)` in [map_info.py](src/map_info.py). The data is downloaded from the website <https://data.geobasis-bb.de/geobasis/daten/dgm/xyz/> and stored as local files under the specified path `path_to_elevation_data` (e.g. the 'elevation' subdirectory).

### Railway Map (OSM)

Store the relevant OSM file locally (e.g. in the `map` subdirectory) and specify `path_to_osm_file` in [data.py](src/data.py).

### Frames

Each frame contains synchronous data from a stereo camera setup and various sensors, where the same frame corresponds to the same filename (i.e. number 000000).:

- Images for each camera (JPG files)
- Poses from RTK-GPS (YAML files)

To avoid having to use ROS directly to be able to interact with the original ROS Bags containing the recorded information, the file [bag_data.py](src/bag_data.py) can be used to directly read ROS messages from relevant topics at indicated timestamps and export them. The idea here is to annotate a selection of images and use their timestamps to export the full information (synchronized pose and stereo image) for each.

### Annotations

Manually created using the tool <https://www.robots.ox.ac.uk/~vgg/software/via/>, by uploading the relevant images and drawing each railway track as a sequence of points. The annotations can be exported as a CSV file, which are read by the `Annotation` class in [annotation.py](src/annotation.py). Finally, specify `path_to_annotations` in [data.py](src/data.py).

## Python Setup

### Python Libraries

See [requirements.txt](src/requirements.txt) or use directly for quick installation:

```console
pip install -r requirements.txt
```

### File Structure

Files to interact with:

- [data.py](src/data.py): Specifies data locations in the file system.
- [main.py](src/main.py): Executes the entire pipeline. Defines the initial guess, camera intrinsics, frames, and other configurations.
- [bag_data.py](src/bag_data.py): Conversion of original ROS bag data into frames that can be used by as inputs.

Data objects:

- [data.py](src/camera.py): Camera class for attributes and specific methods
- [railway.py](src/railway.py): Railway class with processed OSM & elevation data
  - [import_osm.py](src/import_osm.py): Imports OSM data behind the scenes
- [keyframe.py](src/keyframe.py): basic Frame & more sophisticated Keyframe class
  - [annotation.py](src/annotation.py): Annotation class, part of Keyframe
  - [gps.py](src/gps.py): GPS class, part of Keyframe

Static methods:

- [data.py](src/transformation.py): Transformation class
- [visualization.py](src/visualization.py): Visualization class
- [map_info.py](src/map_info.py): MapInfo class for map information retrieval

## C++ Setup

### Pybind11 as Git Submodule

Pull the pybind11 submodule:

```console
git submodule update --init --recursive
```

### Dependencies

- Eigen3 (<http://eigen.tuxfamily.org/index.php?title=Main_Page>)
- Ceres Solver (<http://ceres-solver.org/>)
- OpenCV
- glog
- gflags

Install these packages:

```console
sudo apt install libeigen3-dev
sudo apt install libceres-dev
sudo apt install libopencv-dev
```

(this should install the remaining dependencies automatically)

### C++ Compilation

```console
cd src/cpp/
mkdir build
cd build
cmake ..
```

### Automatic Pybind11 Compilation

This process creates a CPython file, from which Python can access functions. The file will be created in the same directory as the C++ file with the name `optimization.cpython.<cpython_version>.so`

```console
cd src/cpp/build
make
make install
```

Now, when running the Python file (main.py), it should be able to access the C++ functions.

If cpp.optimization is not found by Python, make sure the Pybind11 compilation and the virtual environment are using the same Python versions. If there is another issue, it might be due to missing package dependencies.

### Alternative: Manual Pybind11 Compilation

Make executable, include all dependencies (-I) in their directories and link (-l) the libraries.

The below command is for macOS. For Linux replace "-undefined dynamic_lookup" with "-fPIC". The command assumes the Eigen3, glog, gflags, pybind11 and Python3.9 header files are located in the /usr/local/include/ directory. If this is not the case, specify the correct paths after each -I flag.

```console
cd src/cpp

g++ -std=c++17 -I/usr/local/include/eigen3/ -I/usr/local/include/glog/ -I/usr/local/include/gflags/ -I/usr/local/include/pybind11/ -I/usr/local/include/python3.9 -o optimization -undefined dynamic_lookup $(python3-config --includes) optimization.cc -o optimization$(python3-config --extension-suffix) -lceres
```

More info and/or troubleshooting: <https://pybind11.readthedocs.io/en/latest/compiling.html#building-manually>

## Contact

Author: Eric Tüschenbönner

E-mail: <eric.tueschenboenner@gmail.com>

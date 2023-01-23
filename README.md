# LROD_extrinsics_estimation
**<u>Online Estimation of Camera Extrinsics using Map Information</u>**

The purpose of this project was to create an algorithm to compute the relative pose between GPS and a monocular, intrinsically-calibrated camera at the front of a track vehicle.  Assuming that the monocular camera and GPS are installed on one rigid body, the extrinsics, once computed, can continuously be
used until the camera is dismounted from the train. 

The six parameters of the pose are computed in a multi-staged fashion. After the railway tracks have been successfully detected, the functions computing the six pose estimates are called in a predefined order by the main function.



**Installation**

The following packages are needed to run this project:

- numpy (version 1.23.4)

- matplotlib (version 3.6.2)

- scipy (version 1.9.3)

- pandas (version 1.5.0)

  

**Set-up**

For running this project few set-ups have to be done.

- In the "ImageAttributes"-class defined starting on line 24 in "main.py", the path for the images of the camera and the path for the train poses synced to the image frames must be defined,
- The instrinsics of the camera have to be defined on line 93 in the "main.py",
- From line 132-140 in the "main.py" the image numbers to looked through have to be initialized,
- In line 38 of "YPosition_estimate.py"  and line 47 in "XZPosition_estimate.py" the path to where the downloaded elevation data should be stored, needs to be defined
- In line 48 in "XZPosition_estimate.py" the path to the .csv file containing the pole locations, must be defined.



**Usage**

- After haven gone through the steps defined in the Set-up paragraph, the "main.py" file can be run to start the algorithm.
- In functions "track_information.py", "vanishing_point_detection.py" and "ZRot_estimate.py" the boolean "isImgVisualization" can be set to "True" to see the visualization of the respective parameters. The images, the information is plotted in, only need to be shown/saved.



**Contact**

Author: Nicolina Spiegelhalter

E-mail: spiegeln@ethz.ch


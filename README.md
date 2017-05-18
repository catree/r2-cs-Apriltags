Cornell Cup Robotics
Apriltags Research Project
Nicolas Buitrago

The system has been created in order to be able to run debugging on Apriltag detection system in order to draw comparison to other possible localization systems.
The following steps need to be followed in order to apropriately run the system.
1. Create a virtual drive E:/ that partitions RAM. This will improve performance slightly.
1a. Alternatively the location of all the files can be changed in the main code and a hardrive can be used to store everything.
2. Install opencv dependencies in order to be able to run the main code.
3. Ensure all file locations are correctly called in the main code, the coordinate information file must be in the location E://coordapril.txt this hardcoded into the child process and cannot be changed. If you do not file an E:/ drive it can simply be mounted using IMGdisk.
4. Recommended compiler is Visual studios 2015
5. For accurate results, calibrate your own camera to find its intrinsic camera matrix and change the code in the source .cpp file to fit with your camera matrix. More information on this process can be found at: http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
6. Run!, Output shows distance and angle estimates. 

To print your own tags examples can be found online but the full repository can be found at: https://april.eecs.umich.edu/software/apriltag/
in the pre-generated tag families section.

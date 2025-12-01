# Data

This directory contains the data files used in the project. The files are organized by their respective sources, processing type, and frequency.

## Table description

* time - ISO UTC timestamp of the form YYYY-MM-DD HH:mm:ss.ssss+HH:MM (where the +HH:MM is the timezone offset)
* speed - Speed measurement in meters per second
* bearing - Bearing measurement in degrees
* altitude - Altitude measurement in meters
* longitude - Longitude measurement in degrees
* latitude - Latitude measurement in degrees
* qz - Quaternion component
* qy - Quaternion component
* qx - Quaternion component
* qw - Quaternion component
* roll - Roll angle in degrees
* pitch - Pitch angle in degrees
* yaw - Yaw angle in degrees
* acc_z - Acceleration in the Z direction (meters per second squared)
* acc_y - Acceleration in the Y direction (meters per second squared)
* acc_x - Acceleration in the X direction (meters per second squared)
* gyro_z - Angular velocity around the Z axis (radians per second)
* gyro_y - Angular velocity around the Y axis (radians per second)
* gyro_x - Angular velocity around the X axis (radians per second)
* mag_z - Magnetic field strength in the Z direction (micro teslas)
* mag_y - Magnetic field strength in the Y direction (micro teslas)
* mag_x - Magnetic field strength in the X direction (micro teslas)
* relativeAltitude - Relative altitude measurement (meters)
* pressure - Atmospheric pressure measurement (milli bar)
* grav_z - Gravitational acceleration in the Z direction (meters per second squared)
* grav_y - Gravitational acceleration in the Y direction (meters per second squared)
* grav_x - Gravitational acceleration in the X direction (meters per second squared)

## Structure

- `input/`: Contains the pre-processed input files that are ready for processing as well as an image that highlights the route. Raw recordings are available upon request. They are not stored in version control due to their size.
- `truth/`: Contains the ground truth data files of processed trajectories that are used for validation and testing. These files should be used to compare against alternative processing methods and navigation algorithms.

The following folders contain processed trajectory data files that simulate various GPS conditions that are less than ideal:

- `degraded/`: Contains trajectory data files that simulate degraded GPS conditions (i.e. less accurate GPS data).
- `spoofed/`: Contains trajectory data files that simulate GPS spoofing conditions (i.e. GPS data that has been altered to mislead the navigation system and contains a fixed offset).
- `intermittent/`: Contains trajectory data files that simulate intermittent GPS conditions (i.e. GPS data that is not available for certain periods of time).
- `combo/`: Contains trajectory data files that simulate a combination of degraded, spoofed, and intermittent GPS conditions.

## Use

To test alternative processing methods and navigation algorithms, you should design your navigation algorithm as you see fit to leverage the available data in the input files. For each degraded condition available you should configure your experiment's `GnssDegradationConfiguration` to match the specific characteristics of that condition and process your experiment accordingly.
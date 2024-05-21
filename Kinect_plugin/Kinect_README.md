# Kinect Integration with Robot Arm Simulator

This repository provides the steps to integrate a Kinect sensor into a robot arm simulation using ROS and Gazebo. Follow the instructions below to set up the Kinect, modify the robot URDF, and run the simulation.

## Prerequisites

- ROS (Robot Operating System)
- Gazebo
- catkin workspace
- VS Code (optional, for running the Python script)

## Setup Instructions

### 1. Add the Kinect File to the Package

1. Copy the Kinect file (`kinect.dae`) into the appropriate package directory in your workspace.

### 2. Modify the URDF of the Robot Arm

1. Locate the URDF file for the robot arm within your package.
2. Add the Kinect camera URDF snippet to the robot arm's URDF file and don't forget to adapt it with the required parent link.

    ```xml
    <!-- Add the Kinect sensor -->
    <link name="kinect_link">
        <visual>
            <geometry>
                <mesh filename="package://your_package_name/path_to_kinect.dae" />
            </geometry>
        </visual>
    </link>
    <joint name="kinect_joint" type="fixed">
        <parent link="base_link"/>
        <child link="kinect_link"/>
        <origin xyz="0 0 1" rpy="0 0 0"/>
    </joint>
    ```

### 3. Update the Kinect Path

1. Ensure the path to `kinect.dae` in the URDF file is compatible with your workspace directory structure.
2. Update any other paths for compatibility with your workspace.

### 4. Build the Workspace

Navigate to your catkin workspace and build it:

```sh
catkin build
```

## Run Camera Stream
### 1- Run the launch file of the the robot using 
```sh
source devel/setup.bash

roslaunch pkg_name gazebo.launch
```
### 2- Run the Stream code using stream_sim.py using VScode or 

```sh
rosrun pkg_name stream_sim.py 
note(you have to make the python file executable to use rosrun)
```
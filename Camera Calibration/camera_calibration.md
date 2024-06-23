# Camera Calibration

This project provides tools and scripts for calibrating a camera using a calibration pattern, such as a chessboard or a circle grid. Camera calibration is essential for various computer vision applications, enabling accurate measurements and corrections for lens distortions.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Calibration Process](#calibration-process)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

## Introduction

Camera calibration is the process of estimating the intrinsic parameters of a camera, including focal length, principal point, and distortion coefficients. These parameters are crucial for applications like 3D reconstruction, object tracking, and augmented reality.

This project uses OpenCV to perform camera calibration. It involves capturing multiple images of a known calibration pattern we've used the three known patterns from different angles and then computing the camera's parameters based on these images.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/camera-calibration.git
   cd camera-calibration
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## Usage

Capture Images:

- Capture multiple images of the calibration pattern from different angles. Ensure the entire pattern is visible in each image.
- Save these images in a directory, e.g., calibration_images/.

Run Calibration Script:

- Use the provided calibration script to process the images and compute the camera parameters.

Example usage:
   ```bash
   python calibrate_camera.py --images_dir calibration_images/ --pattern_type chessboard --pattern_size 9x6
   ```
## Calibration Process
1. Load Images:

   - The script loads all images from the specified directory.

2. Find Pattern:

   - For each image, the script detects the corners of the calibration pattern.

3. Compute Calibration:

   - Using the detected corners from all images, the script computes the camera's intrinsic parameters and distortion coefficients.

4. Save Results:

   - The computed parameters are saved to a file for later use

## Results
The calibration script outputs the following parameters:

1. Camera Matrix (Intrinsic Parameters):
   - Includes the focal lengths and the principal point.

2. Distortion Coefficients:
   - Includes the radial and tangential distortion parameters.

3. Rotation and Translation Vectors:
   - Describe the camera's position and orientation relative to the calibration pattern.

These parameters can be used to undistort images, measure objects, and perform various computer vision tasks.
## Troubleshooting

- Pattern Not Found:

    Ensure the entire pattern is visible and not occluded in the images.
    Check if the pattern size and type specified match the actual pattern used.

- High Reprojection Error:

    Ensure that the images cover a wide range of angles and distances.
    Use more images to improve the accuracy of the calibration.

- Installation Issues:

    Verify that all dependencies are installed correctly.
    Check for compatibility issues with your Python and OpenCV versions.

## Acknowledgements
- OpenCV library: https://opencv.org/
- NumPy library: https://numpy.org/
- Calibration pattern images: https://calib.io/pages/camera-calibration-pattern-generator


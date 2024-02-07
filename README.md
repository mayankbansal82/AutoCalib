# Camera Calibration using Zhang's Method

This repository presents an implementation of the camera calibration technique using Zhang's method, a highly regarded method for estimating intrinsic and extrinsic parameters of a camera. The project leverages geometric error minimization to refine image calibration, enhancing the accuracy of 3D computer vision applications.

## Project Overview

Camera calibration is a crucial step in computer vision applications involving 3D geometry. It involves estimating the camera's intrinsic parameters (like focal length and principal point) and distortion coefficients. This implementation follows Zhengyou Zhang's seminal work, providing an automated, efficient, and robust calibration method.

## Features

- **Initial Parameter Estimation**: Estimates initial camera intrinsic matrix, distortion coefficients, and camera extrinsics using a checkerboard calibration target.
- **Geometric Error Minimization**: Employs non-linear optimization to minimize the geometric error, significantly improving calibration accuracy.

## Results

![Before Calibration](/Result/combined_0.png)


![Before Calibration](/Result/combined_1.png)

The calibration process has yielded the following intrinsic and distortion parameters for the camera:

### Distortion Coefficients
- **k1**: `0.0012510872`
- **k2**: `-0.004825358`

### Intrinsic Matrix Parameters
- **alpha** (focal length x-axis): `2056.1066` pixels
- **beta** (focal length y-axis): `2040.5040` pixels
- **gamma** (skew coefficient): `-1.0171`
- **u0** (principal point x-coordinate): `761.6552` pixels
- **v0** (principal point y-coordinate): `1351.3085` pixels

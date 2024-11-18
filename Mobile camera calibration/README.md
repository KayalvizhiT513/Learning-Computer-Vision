# Camera Calibration using OpenCV

This repository contains a Python script for camera calibration using a set of checkerboard images. The calibration process computes the camera's intrinsic and extrinsic parameters, including focal lengths, distortion coefficients, and rotation/translation vectors.

## Features

- **Detects Checkerboard Corners**: Automatically finds and refines chessboard corners in the input images.
- **Camera Calibration**: Computes the camera matrix and distortion coefficients.
- **Reprojection Error Calculation**: Evaluates the accuracy of the calibration process.
- **Resizes Images for Display**: Ensures the images fit within a fixed-size window for visualization.
- **Outputs Key Calibration Parameters**: Includes the camera matrix, distortion coefficients, and focal lengths.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- glob

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KayalvizhiT513/Learning-Computer-Vision.git
   cd Mobile-camera-calibration
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Place Checkerboard Images**  
   Ensure your checkerboard images are in the same directory as the script or provide the correct path.

2. **Run the Calibration Script**  
   Execute the script to calibrate the camera:
   ```bash
   python camera_calibration.py
   ```

3. **View Results**  
   The script will display images with detected corners, and output calibration results including:
   - Camera matrix
   - Distortion coefficients
   - Rotation and translation vectors
   - Average reprojection error
   - Focal lengths (fx, fy)

## Output Example

```
Re-projection Error: 3.7241138747144906
Camera matrix :
[[3.37874415e+03 0.00000000e+00 2.23872717e+03]
 [0.00000000e+00 3.38781549e+03 1.69636687e+03]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
dist :
[[ 0.05987212 -0.08622807 -0.00375573 -0.00087368 -0.33098816]]

...

Number of objects points :  46
Number of objects points :  46
Average error: 0.5426073728370493
Focal Lengths:
  Focal Length (fx): 3378.744149359157 pixels
  Focal Length (fy): 3387.815485753428 pixels
```

## Notes

- The checkerboard should have the same number of rows and columns as defined by the `CHECKERBOARD` variable.
- Images should be captured with the same camera and lens settings for accurate calibration.

---

Happy calibrating! 😊

# SmartVision - Assignment 1

SmartVision is a Python-based real-time computer vision application that integrates multiple interactive modes. It demonstrates image processing, camera calibration, AR overlays, and panorama stitching using OpenCV, NumPy, and Trimesh.

---

## Features

1. **Filters Mode**
   - Apply **Gaussian** or **Bilateral** filters in real-time.
   - Adjust **brightness** and **contrast** using trackbars.
   - Perform **edge detection** with Canny.
   - Compute and display **color histograms**.



2. **Color Spaces Mode**
   - Convert camera feed into **RGB**, **Grayscale**, or **HSV** dynamically.

   

3. **Camera Calibration Mode**
   - Calibrate the camera using a chessboard pattern.
   - Compute camera matrix and distortion coefficients.
   - Required for accurate AR overlay.

   

4. **AR / 3D Overlay Mode**
   - Detect **ArUco markers** and project a **3D OBJ model**.
   - Uses camera calibration for correct pose estimation.
   - Supports skipping frames for better performance.

   

5. **Panorama Mode**
   - ORB-based **feature matching** for stitching two consecutive frames.
   - Allows controlled camera movement for the second frame.

6. **Control**
Mode Switch:
f → Filters
o → Color Spaces
c → Camera Calibration
a → AR / 3D Overlay
p → Panorama
ESC → Quit application

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/assignment_1.git
cd assignment_1

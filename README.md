# SmartVision

## Overview
SmartVision is a Python application for real-time computer vision tasks including:
- Image filtering
- Color analysis
- Camera calibration
- Augmented Reality (AR) using ArUco markers
- Panorama stitching

The app is designed to be interactive, allowing users to switch between modes with simple key presses.

## Features
| Mode       | Key | Description |
|------------|-----|-------------|
| Filters    | f   | Apply Gaussian or Bilateral filters, adjust brightness and contrast, detect edges and lines, display histogram |
| Color      | o   | Convert frame to grayscale and HSV and display side-by-side |
| Calibration| c   | Perform camera calibration using chessboard images |
| AR         | a   | Augmented Reality with 3D models projected on ArUco markers |
| Panorama   | p   | Stitch frames into a panorama in real-time |
| Quit       | ESC | Exit the application |

## Requirements
- Python 3.12.11 recommended

### Dependencies
Install dependencies via pip:
```
numpy>=2.2.6
opencv-python>=4.11.0
opencv-contrib-python>=4.10.0
trimesh>=4.9.0
```
Install via pip:
```
pip install -r requirements.txt
```
Or via Conda (recommended):
```
conda create -n py312 python=3.12.11 (Recommended)
conda activate py312
pip install -r requirements.txt
```

## Project Structure
```
SmartVision/
├─ app.py               # Main application
├─ assets/
│  └─ trex_model.obj    # 3D model used in AR
├─ camera_calib.npz     # Saved calibration file (optional)
├─ requirements.txt     # Python dependencies
└─ README.md
```
Make sure the `assets/trex_model.obj` file is included for AR mode.

## How to Run
1. Clone or download the project:
```
git clone <your-repo-url>
cd SmartVision
```
2. Activate environment (if using Conda):
```
conda activate py312
```
3. Install dependencies (if not already installed):
```
pip install -r requirements.txt
```
4. Run the app:
```
python app.py
```
5. Interact with the app:
- Press one of the mode keys: `[f/c/a/p/o]`
- Press `ESC` to quit the application.

## Notes
- Ensure your camera is connected before starting the app.
- If `camera_calib.npz` exists, calibration will be loaded automatically. Otherwise, follow the calibration mode (c) to generate a new calibration file.
- AR mode requires the 3D object file (`trex_model.obj`) and successful calibration.
- Panorama mode may require multiple frames to stitch properly.
- Deprecation warnings from NumPy (v2.25+) regarding array-to-int conversion are normal and do not affect functionality.

## License
This project is open-source and practicing purpose. Feel free to modify and use it for educational or personal purposes.


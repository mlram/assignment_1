import cv2
import numpy as np
import trimesh

# Settings
chessboard_size = (9, 6)
obj_file_path = "assets/trex_model.obj"
aruco_marker_length = 0.1  # meters
camera_matrix = None
dist_coeffs = None

# State Variables
mode = 'f'  # f=Filters, c=Calibration, a=AR, p=Panorama
calib_images = []
calib_done = False
ar_mesh = None
prev_frame = None
prev_offset = 0
panorama_canvas = None
ar_skip_frame = 0  # Skip AR frame counter for performance

# Helper functions


def adjust_brightness_contrast(frame, brightness=0, contrast=1.0):
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)


# Color Space Conversion
def convert_color(frame, mode):
    if mode == 0:  # RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif mode == 1:  # GRAY
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif mode == 2:  # HSV
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return frame


def apply_filter(frame, filter_type="None", param1=5, param2=75):
    if filter_type == "Gaussian":
        if param1 % 2 == 0:
            param1 += 1
        return cv2.GaussianBlur(frame, (param1, param1), 0)
    elif filter_type == "Bilateral":
        return cv2.bilateralFilter(frame, param1, param2, param2)
    return frame

# Edge Detection and Line Drawing


def draw_lines(frame, edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                            minLineLength=50, maxLineGap=10)
    line_img = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return line_img


def compute_histogram(frame):
    colors = ('b', 'g', 'r')
    hist_img = np.zeros((300, 256, 3), dtype=np.uint8)
    for i, col in enumerate(colors):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
        for j in range(1, 256):
            h1 = int(hist[j-1][0])
            h2 = int(hist[j][0])
            cv2.line(hist_img, (j-1, 300-h1), (j, 300-h2),
                     (255 if col == 'b' else 0, 255 if col == 'g' else 0, 255 if col == 'r' else 0), 1)
    return hist_img

# Panorama function


def create_panorama(new_frame, prev_frame, panorama_canvas):
    if prev_frame is None:
        h, w = new_frame.shape[:2]
        panorama_canvas = np.zeros((h, w*5, 3), dtype=np.uint8)
        panorama_canvas[:, :w] = new_frame
        return panorama_canvas, new_frame
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(prev_frame, None)
    kp2, des2 = orb.detectAndCompute(new_frame, None)
    if des1 is None or des2 is None:
        return panorama_canvas, new_frame
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 4:
        return panorama_canvas, new_frame
    matches = sorted(matches, key=lambda x: x.distance)
    dx = int(np.median([kp2[m.trainIdx].pt[0] -
             kp1[m.queryIdx].pt[0] for m in matches]))
    h, w = new_frame.shape[:2]
    if panorama_canvas is None:
        panorama_canvas = np.zeros((h, w*5, 3), dtype=np.uint8)
        panorama_canvas[:h, :w] = new_frame
        return panorama_canvas, new_frame
    offset_x = np.max(np.where(panorama_canvas.sum(axis=2) > 0)[1])+1
    if offset_x+dx+w < panorama_canvas.shape[1]:
        panorama_canvas[:h, offset_x+dx:offset_x+dx+w] = new_frame
    return panorama_canvas, new_frame


# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Trackbars
cv2.namedWindow("SmartVision")
cv2.createTrackbar("Brightness", "SmartVision", 100, 200, lambda x: None)
cv2.createTrackbar("Contrast", "SmartVision", 10, 30, lambda x: None)
cv2.createTrackbar("Gaussian Kernel", "SmartVision", 5, 15, lambda x: None)
cv2.createTrackbar("Bilateral Param", "SmartVision", 75, 200, lambda x: None)
cv2.createTrackbar("Canny Thresh1", "SmartVision", 100, 500, lambda x: None)
cv2.createTrackbar("Canny Thresh2", "SmartVision", 200, 500, lambda x: None)
cv2.createTrackbar("Filter Type", "SmartVision", 0, 1, lambda x: None)

filter_map = {0: "Gaussian", 1: "Bilateral"}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, (640, int(frame.shape[0]*640/frame.shape[1])))

    brightness_val = cv2.getTrackbarPos("Brightness", "SmartVision")-100
    contrast_val = cv2.getTrackbarPos("Contrast", "SmartVision")/10
    gauss_kernel = cv2.getTrackbarPos("Gaussian Kernel", "SmartVision")
    bilateral_param = cv2.getTrackbarPos("Bilateral Param", "SmartVision")
    canny_thresh1 = cv2.getTrackbarPos("Canny Thresh1", "SmartVision")
    canny_thresh2 = cv2.getTrackbarPos("Canny Thresh2", "SmartVision")
    filter_type = filter_map[cv2.getTrackbarPos("Filter Type", "SmartVision")]
    color_mode = cv2.getTrackbarPos("Color Mode", "SmartVision")
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key in [ord('f'), ord('c'), ord('a'), ord('p'), ord('o')]:
        mode = chr(key)
    guide_text = "Mode:[f]Filters [o]Color [c]Calib [a]AR/3D [p]Panorama ESC=Quit"

    # Color Filters
    if mode == 'f':
        proc = adjust_brightness_contrast(img, brightness_val, contrast_val)
        proc = apply_filter(
            proc, filter_type, gauss_kernel if filter_type == "Gaussian" else bilateral_param)
        gray_frame = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, canny_thresh1, canny_thresh2)
        proc_lines = draw_lines(proc, edges)
        hist_img = compute_histogram(proc)
        combined = cv2.vconcat(
            [proc_lines, cv2.resize(hist_img, (proc_lines.shape[1], 300))])
        cv2.putText(combined, guide_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("SmartVision", combined)

    # Color Spaces
    elif mode == 'o':
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        combined = np.hstack([img, gray_frame[:, :, None], hsv_frame])
        cv2.putText(combined, guide_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("SmartVision", combined)

    # Camera Calibration
    elif mode == 'c':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret_cb, corners = cv2.findChessboardCorners(
            gray, chessboard_size, None)
        if ret_cb and len(calib_images) < 20:
            calib_images.append(gray)
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret_cb)
        cv2.putText(img, f"Frames:{len(calib_images)}/20",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, guide_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("SmartVision", img)
        if len(calib_images) >= 15 and not calib_done:
            objp = np.zeros(
                (chessboard_size[1]*chessboard_size[0], 3), np.float32)
            objp[:, :2] = np.mgrid[0:chessboard_size[0],
                                   0:chessboard_size[1]].T.reshape(-1, 2)
            objpoints, imgpoints = [], []
            for g in calib_images:
                ret, c = cv2.findChessboardCorners(g, chessboard_size, None)
                if ret:
                    objpoints.append(objp)
                    imgpoints.append(c)
            ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
            calib_done = True
            print("Calibration Done!")

    # AR/3D Overlay
    elif mode == 'a' and calib_done:
        ar_skip_frame += 1
        if ar_skip_frame % 2 != 0:
            cv2.imshow("SmartVision", img)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params)
        proc = img.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(proc, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, aruco_marker_length, camera_matrix, dist_coeffs)
            if ar_mesh is None:
                ar_mesh = trimesh.load(obj_file_path, force='mesh')
                if isinstance(ar_mesh, trimesh.Scene):
                    first_name = list(ar_mesh.geometry.keys())[0]
                    ar_mesh = ar_mesh.geometry[first_name]

                ar_mesh.apply_scale(5.0)
            verts = np.array(ar_mesh.vertices)
            faces = np.array(ar_mesh.faces)
            verts_proj, _ = cv2.projectPoints(
                verts, rvecs[0], tvecs[0], camera_matrix, dist_coeffs)
            verts_proj = verts_proj.reshape(-1, 2).astype(int)
            for f in faces:
                pts = verts_proj[f]
                cv2.polylines(proc, [pts], True, (0, 255, 0), 1)
        cv2.putText(proc, guide_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("SmartVision", proc)

    # Panorama
    elif mode == 'p':
        panorama_canvas, prev_frame = create_panorama(
            img, prev_frame, panorama_canvas)
        if panorama_canvas is not None:
            cv2.putText(panorama_canvas, guide_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("SmartVision", panorama_canvas)

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import trimesh
import time
import os

# ==============================
# CONFIGURATION
# ==============================
CHESSBOARD_SIZE = (9, 6)
ARUCO_MARKER_LENGTH = 0.1  # meters
OBJ_FILE_PATH = "assets/trex_model.obj"
CALIB_FILE = "camera_calib.npz"

# Global state
camera_matrix = None
dist_coeffs = None
calib_done = False
mode = "f"  # f=Filters, c=Calibration, a=AR, p=Panorama, o=Color
calib_images = []
ar_mesh = None
prev_frame = None
panorama_canvas = None
frame_skip = 0


# FUNCTIONS


def adjust_brightness_contrast(frame, brightness=0, contrast=1.0):
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)


def apply_filter(frame, f_type="Gaussian", p1=5, p2=75):
    if f_type == "Gaussian":
        if p1 % 2 == 0:
            p1 += 1
        return cv2.GaussianBlur(frame, (p1, p1), 0)
    elif f_type == "Bilateral":
        return cv2.bilateralFilter(frame, p1, p2, p2)
    return frame


def draw_lines(frame, edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                            minLineLength=50, maxLineGap=10)
    line_img = frame.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return line_img


def compute_histogram(frame):
    colors = ('b', 'g', 'r')
    hist_img = np.zeros((300, 256, 3), dtype=np.uint8)
    for i, col in enumerate(colors):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
        for j in range(1, 256):
            cv2.line(hist_img,
                     (j-1, 300 - int(hist[j-1].item())),
                     (j, 300 - int(hist[j].item())),
                     (255 if col == 'b' else 0,
                      255 if col == 'g' else 0,
                      255 if col == 'r' else 0),
                     1)
    return hist_img


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

    offset_x = np.max(np.where(panorama_canvas.sum(axis=2) > 0)[1]) + 1
    if offset_x + dx + w < panorama_canvas.shape[1]:
        panorama_canvas[:h, offset_x+dx:offset_x+dx+w] = new_frame

    return panorama_canvas, new_frame


def save_calibration(mtx, dist):
    np.savez(CALIB_FILE, mtx=mtx, dist=dist)
    print("💾 Calibration saved.")


def load_calibration():
    global camera_matrix, dist_coeffs, calib_done
    if os.path.exists(CALIB_FILE):
        data = np.load(CALIB_FILE)
        camera_matrix = data['mtx']
        dist_coeffs = data['dist']
        calib_done = True
        print("Calibration loaded from file.")


# MAIN APP


def main():
    global mode, calib_done, camera_matrix, dist_coeffs, ar_mesh, prev_frame, panorama_canvas, frame_skip

    load_calibration()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    cv2.namedWindow("SmartVision")
    cv2.createTrackbar("Brightness", "SmartVision", 100, 200, lambda x: None)
    cv2.createTrackbar("Contrast", "SmartVision", 10, 30, lambda x: None)
    cv2.createTrackbar("Gaussian", "SmartVision", 5, 15, lambda x: None)
    cv2.createTrackbar("Bilateral", "SmartVision", 75, 200, lambda x: None)
    cv2.createTrackbar("Canny1", "SmartVision", 100, 500, lambda x: None)
    cv2.createTrackbar("Canny2", "SmartVision", 200, 500, lambda x: None)
    cv2.createTrackbar("FilterType", "SmartVision", 0, 1, lambda x: None)

    filter_map = {0: "Gaussian", 1: "Bilateral"}
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()

    prev_time = time.time()
    print("SmartVision ready. Press [f/c/a/p/o], ESC=Quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not captured.")
            break

        img = cv2.resize(frame, (640, int(frame.shape[0]*640/frame.shape[1])))
        guide = "Mode:[f]Filters [o]Color [c]Calib [a]AR [p]Panorama  ESC=Quit"

        # Trackbar values
        brightness = cv2.getTrackbarPos("Brightness", "SmartVision") - 100
        contrast = cv2.getTrackbarPos("Contrast", "SmartVision") / 10
        gk = cv2.getTrackbarPos("Gaussian", "SmartVision")
        bp = cv2.getTrackbarPos("Bilateral", "SmartVision")
        ct1 = cv2.getTrackbarPos("Canny1", "SmartVision")
        ct2 = cv2.getTrackbarPos("Canny2", "SmartVision")
        filter_type = filter_map[cv2.getTrackbarPos(
            "FilterType", "SmartVision")]

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Exiting...")
            break
        if key in [ord('f'), ord('c'), ord('a'), ord('p'), ord('o')]:
            mode = chr(key)
            print(f"Switched mode: {mode}")

        # FPS
        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        cv2.putText(img, f"FPS: {fps:.1f}", (500, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # =============== FILTER MODE ===============
        if mode == 'f':
            proc = adjust_brightness_contrast(img, brightness, contrast)
            proc = apply_filter(proc, filter_type,
                                gk if filter_type == "Gaussian" else bp)
            gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, ct1, ct2)
            proc_lines = draw_lines(proc, edges)
            hist_img = compute_histogram(proc)
            combined = cv2.vconcat(
                [proc_lines, cv2.resize(hist_img, (proc_lines.shape[1], 300))])
            cv2.putText(combined, guide, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow("SmartVision", combined)

        # =============== COLOR MODE ===============
        elif mode == 'o':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            combined = np.hstack(
                [img, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), hsv])
            cv2.putText(combined, guide, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow("SmartVision", combined)

        # =============== CALIBRATION MODE ===============
        elif mode == 'c':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret_cb, corners = cv2.findChessboardCorners(
                gray, CHESSBOARD_SIZE, None)
            if ret_cb and len(calib_images) < 20:
                calib_images.append(gray)
                cv2.drawChessboardCorners(
                    img, CHESSBOARD_SIZE, corners, ret_cb)
            cv2.putText(img, f"Frames:{len(calib_images)}/20", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(calib_images) >= 15 and not calib_done:
                print("📸 Performing calibration...")
                objp = np.zeros(
                    (CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                                       0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
                objpoints, imgpoints = [], []
                for g in calib_images:
                    ret, c = cv2.findChessboardCorners(
                        g, CHESSBOARD_SIZE, None)
                    if ret:
                        objpoints.append(objp)
                        imgpoints.append(c)
                _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None)
                calib_done = True
                save_calibration(camera_matrix, dist_coeffs)
                print("Calibration complete.")
            cv2.putText(img, guide, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow("SmartVision", img)

        # =============== AR MODE ===============
        elif mode == 'a' and calib_done:
            frame_skip += 1
            if frame_skip % 2 != 0:
                cv2.imshow("SmartVision", img)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=aruco_params)
            proc = img.copy()
            status_text = "AR: Marker Not Found"
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(proc, corners, ids)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, ARUCO_MARKER_LENGTH, camera_matrix, dist_coeffs)
                status_text = "AR: Marker Found "
                if ar_mesh is None:
                    ar_mesh = trimesh.load(OBJ_FILE_PATH, force='mesh')
                    if isinstance(ar_mesh, trimesh.Scene):
                        first = list(ar_mesh.geometry.keys())[0]
                        ar_mesh = ar_mesh.geometry[first]
                    ar_mesh.apply_scale(5.0)
                verts = np.array(ar_mesh.vertices)
                faces = np.array(ar_mesh.faces)
                verts_proj, _ = cv2.projectPoints(
                    verts, rvecs[0], tvecs[0], camera_matrix, dist_coeffs)
                verts_proj = verts_proj.reshape(-1, 2)
                verts_proj = verts_proj[~np.isnan(verts_proj).any(axis=1)]
                verts_proj = verts_proj.astype(int)
                for f in faces:
                    pts = verts_proj[f]
                    if pts.shape[0] == 2:  # skip invalid faces
                        continue
                    cv2.polylines(proc, [pts], True, (0, 255, 0), 1)
            cv2.putText(proc, status_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(proc, guide, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow("SmartVision", proc)

        # =============== PANORAMA MODE ===============
        elif mode == 'p':
            panorama_canvas, prev_frame = create_panorama(
                img, prev_frame, panorama_canvas)
            if panorama_canvas is not None:
                cv2.putText(panorama_canvas, guide, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.imshow("SmartVision", panorama_canvas)

    cap.release()
    cv2.destroyAllWindows()
    print("SmartVision closed cleanly.")


if __name__ == "__main__":
    main()

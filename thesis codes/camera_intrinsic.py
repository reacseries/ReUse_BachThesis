import numpy as np
import cv2
import glob
import os

# === SETTINGS ===
CHECKERBOARD = (7, 5)
SQUARE_SIZE = 20.0  # mm

# === Your folder path ===
FOLDER = r"C:\Users\lokadm\realsense cam\captures"

# Prepare 3D object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Load all detected corners
images = sorted(glob.glob(os.path.join(FOLDER, "img_*.png")))
corners_list = sorted(glob.glob(os.path.join(FOLDER, "corners_*.npy")))

objpoints = []
imgpoints = []

for cfile in corners_list:
    corners = np.load(cfile)
    if corners is not None:
        objpoints.append(objp)
        imgpoints.append(corners)

# Calibrate camera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (1280, 720), None, None
)

# Save results
np.savez("../camera_calibration.npz",
         cameraMatrix=K, distCoeffs=dist,
         rvecs=rvecs, tvecs=tvecs)

print("✅ Saved full calibration with rvecs & tvecs to 'camera_calibration.npz'")
print("RMS reprojection error:", ret)

import numpy as np
import cv2
import csv

# load camera calibration
data = np.load("../camera_calibration.npz", allow_pickle=True)
rvecs = data["rvecs"]
tvecs = data["tvecs"]

# load robot poses (each row = XYZRxRyRz in mm and degrees)
robot_T = []
with open("../posses.csv") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        x, y, z, rx, ry, rz = map(float, row)
        # convert to rotation matrix
        Rx = np.deg2rad(rx)
        Ry = np.deg2rad(ry)
        Rz = np.deg2rad(rz)
        # assume XYZ order (adjust if your robot uses ZYX, etc.)
        R, _ = cv2.Rodrigues(np.array([Rx, Ry, Rz]))
        t = np.array([[x], [y], [z]]) / 1000.0  # convert mm → m
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:] = t
        robot_T.append(T)

# convert rvecs/tvecs into same form (target→camera)
R_target2cam = [cv2.Rodrigues(r)[0] for r in rvecs]
t_target2cam = [t.reshape(3,1)/1000.0 for t in tvecs]  # mm → m

# compute robot motion (flange→base)
R_gripper2base, t_gripper2base = [], []
for i in range(1, len(robot_T)):
    T_prev = robot_T[i-1]
    T_curr = robot_T[i]
    T_delta = np.linalg.inv(T_prev) @ T_curr
    R_gripper2base.append(T_delta[:3,:3])
    t_gripper2base.append(T_delta[:3,3:])

# compute target (checkerboard) motion in camera
R_target2cam_delta, t_target2cam_delta = [], []
for i in range(1, len(R_target2cam)):
    R_prev, t_prev = R_target2cam[i-1], t_target2cam[i-1]
    R_curr, t_curr = R_target2cam[i], t_target2cam[i]
    T_prev = np.eye(4); T_prev[:3,:3]=R_prev; T_prev[:3,3]=t_prev.ravel()
    T_curr = np.eye(4); T_curr[:3,:3]=R_curr; T_curr[:3,3]=t_curr.ravel()
    T_delta = np.linalg.inv(T_curr) @ T_prev
    R_target2cam_delta.append(T_delta[:3,:3])
    t_target2cam_delta.append(T_delta[:3,3])

# run OpenCV hand-eye calibration
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam_delta, t_target2cam_delta,
    method=cv2.CALIB_HAND_EYE_TSAI)

print("\n=== HAND–EYE CALIBRATION RESULT ===")
print("R_cam2gripper:\n", R_cam2gripper)
print("t_cam2gripper (m):\n", t_cam2gripper.ravel())
np.savez("../handeye_result.npz", R=R_cam2gripper, t=t_cam2gripper)
print("Saved to handeye_result.npz")

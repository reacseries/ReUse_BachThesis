from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
import numpy as np

# ===============================
# Helpers
# ===============================
def normalize_0_180(deg: float) -> float:
    """Map any angle (deg) to [0, 180)."""
    deg = deg % 180.0
    if deg < 0:
        deg += 180.0
    return deg

def rotz(deg: float) -> np.ndarray:
    """Rotation about Z by deg (right-handed, counter-clockwise, standard math)."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)

# ===============================
# Transform: Camera → Gripper
# ===============================
R_gc_default = np.array([
    [0, -1, 0],
    [0,  0, 1],
    [-1, 0, 0]
], dtype=float)
t_gc_default = np.array([0.085, -0.220, 0.040], dtype=float)  # meters

def tf_camera_to_gripper_point(point_cam,
                               R_gc=R_gc_default,
                               t_gc=t_gc_default):
    """Transform a 3D point (camera frame -> gripper frame)."""
    p = np.asarray(point_cam, dtype=float).reshape(3, 1)
    R = np.asarray(R_gc, dtype=float).reshape(3, 3)
    t = np.asarray(t_gc, dtype=float).reshape(3, 1)
    pg = R @ p + t
    return pg.flatten()

def tf_camera_to_gripper_dir(dir_cam,
                             R_gc=R_gc_default):
    """Transform a direction (no translation)."""
    v = np.asarray(dir_cam, dtype=float).reshape(3, 1)
    R = np.asarray(R_gc, dtype=float).reshape(3, 3)
    vg = R @ v
    return vg.flatten()

# ===============================
# Angle utilities
# ===============================
def axis_unit_vector_in_camera(r_rad: float, w: float, h: float) -> np.ndarray:
    """
    Build the major-axis direction vector in the camera frame from OBB angle r.
    If the YOLO OBB rotation is along the minor axis (h > w),
    we rotate by 90° (π/2) to align it with the longer edge.
    """
    # ✅ Fix: align with major axis
    if h > w:
        r_rad += np.pi / 2.0
    return np.array([np.cos(r_rad), np.sin(r_rad), 0.0], dtype=float)

def angle_from_vector_xy(v: np.ndarray) -> float:
    """Angle (radians) from +X using atan2(vy, vx)."""
    vx, vy = v[0], v[1]
    return np.arctan2(vy, vx)

# ===============================
# YOLO + RealSense OBB pipeline
# ===============================
def detection_xyz_and_angles(model, color_frame, depth_frame, intrinsics, img_width, img_height,
                             R_gc=R_gc_default, t_gc=t_gc_default,
                             robot_Rz_deg=None, **yolo_args):
    color_image = np.asanyarray(color_frame.get_data())
    results = model(color_image, **yolo_args)[0]

    detections = []
    if hasattr(results, "obb") and results.obb is not None and len(results.obb.cls) > 0:
        obb = results.obb
        for i in range(len(obb.cls)):
            cls = int(obb.cls[i])
            conf = float(obb.conf[i])

            # Get OBB parameters
            poly = obb.xyxyxyxy[i].detach().cpu().numpy().astype(int)
            x, y, w, h, r = obb.xywhr[i].detach().cpu().numpy().astype(float)

            # Center
            center = np.mean(poly, axis=0).astype(int)
            cx, cy = int(np.clip(center[0], 0, img_width - 1)), int(np.clip(center[1], 0, img_height - 1))

            # Depth (m)
            depth = depth_frame.get_distance(cx, cy)
            if depth <= 0:
                continue
            p_cam = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
            p_grip = tf_camera_to_gripper_point(p_cam, R_gc=R_gc, t_gc=t_gc)

            # --- Orientation handling ---
            # 1️⃣ Major-axis direction in CAMERA frame
            v_cam = axis_unit_vector_in_camera(r_rad=float(r), w=w, h=h)

            # 2️⃣ Angle in camera frame (relative to image base)
            angle_cam_rad = angle_from_vector_xy(v_cam)
            angle_cam_deg = normalize_0_180(np.degrees(angle_cam_rad))

            # 3️⃣ Transform direction to GRIPPER frame
            v_grip = tf_camera_to_gripper_dir(v_cam, R_gc=R_gc)
            angle_grip_rad = angle_from_vector_xy(v_grip)
            angle_grip_deg = normalize_0_180(np.degrees(angle_grip_rad))

            # 4️⃣ Optional: Transform to ROBOT BASE
            angle_base_rad, angle_base_deg = None, None
            if robot_Rz_deg is not None:
                R_bg = rotz(float(robot_Rz_deg))
                v_base = (R_bg @ v_grip.reshape(3,)).reshape(3,)
                angle_base_rad = angle_from_vector_xy(v_base)
                angle_base_deg = normalize_0_180(np.degrees(angle_base_rad))

            detections.append({
                "class_id": cls,
                "class_name": model.names[cls] if hasattr(model, "names") else str(cls),
                "confidence": conf,
                "center_2d": [cx, cy],
                "xyz": p_cam,
                "xyz_gripper": p_grip,
                "poly": poly,
                "r_rad": float(r),
                "angle_cam_rad": angle_cam_rad,
                "angle_cam_deg": angle_cam_deg,
                "angle_grip_rad": angle_grip_rad,
                "angle_grip_deg": angle_grip_deg,
                "angle_base_rad": angle_base_rad,
                "angle_base_deg": angle_base_deg
            })

    return detections, color_image

# ===============================
# Drawing function
# ===============================
def draw_detections(color_image, detections, draw_line_len_px=60):
    out = color_image.copy()
    for det in detections:
        poly = det["poly"]
        cx, cy = det["center_2d"]
        cls_name = det["class_name"]
        conf = det["confidence"]
        xyz = det["xyz"]
        xyz_g = det["xyz_gripper"]
        angle_cam = det["angle_cam_rad"]

        # Draw OBB polygon
        cv2.polylines(out, [poly.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw center and major-axis line
        cv2.circle(out, (cx, cy), 3, (255, 0, 0), -1)
        dx = draw_line_len_px * np.cos(angle_cam)
        dy = draw_line_len_px * np.sin(angle_cam)
        x1, y1 = int(cx - dx), int(cy - dy)
        x2, y2 = int(cx + dx), int(cy + dy)
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Labels
        cv2.putText(out, f"{cls_name} {conf:.2f}", (cx - 40, cy - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        cv2.putText(out, f"Cam: X:{xyz[0]:.3f} Y:{xyz[1]:.3f} Z:{xyz[2]:.3f}",
                    (cx - 90, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(out, f"Grip: X:{xyz_g[0]:.3f} Y:{xyz_g[1]:.3f} Z:{xyz_g[2]:.3f}",
                    (cx - 90, cy + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Angles
        line2 = f"Cam:{det['angle_cam_deg']:.1f}°  Grip:{det['angle_grip_deg']:.1f}°"
        if det["angle_base_deg"] is not None:
            line2 += f"  Base:{det['angle_base_deg']:.1f}°"
        cv2.putText(out, line2, (cx - 90, cy + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    MODEL_PATH = "latestmodel.pt0"
    ROBOT_RZ_DEG = -90.0

    R_gc = R_gc_default
    t_gc = t_gc_default

    model = YOLO(MODEL_PATH)

    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    img_width, img_height = intrinsics.width, intrinsics.height

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            detections, color_image = detection_xyz_and_angles(
                model, color_frame, depth_frame, intrinsics, img_width, img_height,
                R_gc=R_gc, t_gc=t_gc, robot_Rz_deg=ROBOT_RZ_DEG,
                conf=0.5, iou=0.5, stream=False
            )

            vis = draw_detections(color_image, detections)
            cv2.imshow("YOLO OBB: Camera/Gripper/Robot Angles + XYZ", vis)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

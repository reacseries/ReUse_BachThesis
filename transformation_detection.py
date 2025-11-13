from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
import numpy as np
from collections import Counter


# -------------------------------
# Transform: Camera → Gripper
# -------------------------------
def tf_camera_to_gripper(point_cam,
                         R_gc=np.array([
                             [0, -1, 0],
                             [0,  0, 1],
                             [-1, 0, 0]
                         ]),
                         t_gc=np.array([0.085, -0.220, 0.040])):
    """Transform a 3D point from the camera frame to the gripper frame."""
    point_cam = np.array(point_cam).reshape(3, 1)
    R_gc = np.array(R_gc).reshape(3, 3)
    t_gc = np.array(t_gc).reshape(3, 1)
    point_gripper = R_gc @ point_cam + t_gc
    return point_gripper.flatten()


# -------------------------------
# Detection function
# -------------------------------
def detection_xyz(model, color_frame, depth_frame, intrinsics, img_width, img_height, **yolo_args):
    color_image = np.asanyarray(color_frame.get_data())
    results = model(color_image, **yolo_args)[0]

    detections = []
    if results.obb:
        for obb in results.obb:
            cls = int(obb.cls[0])
            conf = float(obb.conf[0])

            # OBB vertices and center
            xyxy = obb.xyxyxyxy[0].cpu().numpy().astype(int)
            center = np.mean(xyxy, axis=0).astype(int)
            cx, cy = center[0], center[1]
            cx = np.clip(cx, 0, img_width - 1)
            cy = np.clip(cy, 0, img_height - 1)

            # Depth (m)
            depth = depth_frame.get_distance(cx, cy)
            if depth <= 0:
                continue

            # 3D point in camera frame
            point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)

            # Transform to gripper frame
            point_3d_gripper = tf_camera_to_gripper(point_3d)

            detections.append({
                "class_id": cls,
                "class_name": model.names[cls],
                "confidence": conf,
                "center_2d": [cx, cy],
                "xyz": point_3d,
                "xyz_gripper": point_3d_gripper,
                "xyxy": xyxy
            })

    return detections, color_image


# -------------------------------
# Drawing function
# -------------------------------
def draw_detection(color_image, detections):
    for det in detections:
        xyxy = det["xyxy"]
        cls_name = det["class_name"]
        conf = det["confidence"]
        xyz = det["xyz"]
        xyz_g = det["xyz_gripper"]

        # Draw oriented bounding box
        cv2.polylines(color_image, [xyxy.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
        cx, cy = det["center_2d"]

        # Labels
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(color_image, label, (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Coordinates in camera and gripper frame
        cv2.putText(color_image,
                    f"Cam: X:{xyz[0]:.3f} Y:{xyz[1]:.3f} Z:{xyz[2]:.3f}",
                    (cx - 70, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.putText(color_image,
                    f"Grip: X:{xyz_g[0]:.3f} Y:{xyz_g[1]:.3f} Z:{xyz_g[2]:.3f}",
                    (cx - 70, cy + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return color_image


# -------------------------------
# Main loop
# -------------------------------
if __name__ == "__main__":
    # Load YOLO OBB model
    model = YOLO("epoch85.pt")

    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # Get camera intrinsics
    align_to = rs.stream.color
    align = rs.align(align_to)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    img_width, img_height = intrinsics.width, intrinsics.height

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            detections, color_image = detection_xyz(
                model, color_frame, depth_frame, intrinsics, img_width, img_height, conf=0.5, iou=0.5, stream=False
            )

            color_image = draw_detection(color_image, detections)

            cv2.imshow("YOLO OBB Detection (Camera + Gripper coords)", color_image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

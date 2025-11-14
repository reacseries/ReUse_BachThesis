from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
import numpy as np
from opcua import Client, ua
import time


def tf_camera_to_gripper(point_cam,
                         R_gc=np.array([[0, -1, 0],
                                        [0, 0, 1],
                                        [-1, 0, 0]]),
                         t_gc=np.array([0.085, -0.220, 0.040])):
    point_cam = np.array(point_cam).reshape(3, 1)
    R_gc = np.array(R_gc).reshape(3, 3)
    t_gc = np.array(t_gc).reshape(3, 1)
    return (R_gc @ point_cam + t_gc).flatten()


def detection_xyz(model, color_frame, depth_frame, intrinsics, img_width, img_height, **yolo_args):
    color_image = np.asanyarray(color_frame.get_data())
    results = model(color_image, **yolo_args)[0]
    detections = []

    if results.obb:
        for obb in results.obb:
            cls = int(obb.cls[0])
            conf = float(obb.conf[0])
            xyxy = obb.xyxyxyxy[0].cpu().numpy().astype(int)

            center = np.mean(xyxy, axis=0).astype(int)
            cx, cy = np.clip(center[0], 0, img_width - 1), np.clip(center[1], 0, img_height - 1)

            depth = depth_frame.get_distance(cx, cy)
            if depth <= 0:
                continue

            point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
            point_3d_gripper = tf_camera_to_gripper(point_3d)

            detections.append({
                "class_name": model.names[cls],
                "confidence": conf,
                "center_2d": [cx, cy],
                "xyz_gripper": point_3d_gripper,
                "xyxy": xyxy
            })
    return detections, color_image


def draw_detection(color_image, detections):
    for det in detections:
        xyxy = det["xyxy"]
        cls_name = det["class_name"]
        conf = det["confidence"]
        xyz_g = det["xyz_gripper"]

        cv2.polylines(color_image, [xyxy.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
        cx, cy = det["center_2d"]
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(color_image, label, (cx - 30, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(color_image,
                    f"X:{xyz_g[0]*1000:.1f} Z:{xyz_g[2]*1000:.1f} mm",
                    (cx - 70, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return color_image


def setup_opcua_client(endpoint="opc.tcp://192.168.0.1:4840"):
    client = Client(endpoint)
    client.connect()
    print(f"[OPC UA] Connected to {endpoint}")

    # ✅ Node IDs from UaExpert
    node_x = client.get_node("ns=4;i=2")  # X variable
    node_z = client.get_node("ns=4;i=3")  # Z variable
    # Optional: test write
    try:
        node_x.set_value(ua.Variant(0.0, ua.VariantType.Float))
        node_z.set_value(ua.Variant(0.0, ua.VariantType.Float))
        print("[OPC UA] Test write succeeded.")
    except Exception as e:
        print("[OPC UA] ⚠️ Test write failed:", e)

    return client, node_x, node_z


if __name__ == "__main__":
    model = YOLO("epoch85.pt")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    img_width, img_height = intrinsics.width, intrinsics.height

    client, node_x, node_z = setup_opcua_client("opc.tcp://192.168.0.1:4840")

    print("[SYSTEM] Realtime detection and OPC UA streaming started... Press ESC to stop.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            detections, color_image = detection_xyz(model, color_frame, depth_frame,
                                                    intrinsics, img_width, img_height, conf=0.5)

            if detections:
                xyz_g = detections[0]["xyz_gripper"]
                xg, zg = float(xyz_g[0] * 1000), float(xyz_g[2] * 1000)

                try:
                    node_x.set_value(ua.Variant(xg, ua.VariantType.Float))
                    node_z.set_value(ua.Variant(zg, ua.VariantType.Float))
                    print(f"[OPC UA] Updated → X={xg:.1f} mm | Z={zg:.1f} mm")
                except Exception as e:
                    print("[OPC UA] Write error:", e)

                time.sleep(0.1)  # small delay

            color_image = draw_detection(color_image, detections)
            cv2.imshow("YOLO + RealSense + OPC UA", color_image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        pipeline.stop()
        client.disconnect()
        cv2.destroyAllWindows()
        print("[CLEANUP] Disconnected and stopped.")

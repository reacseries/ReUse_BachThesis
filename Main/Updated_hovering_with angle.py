###############################################
# 1) IMPORTS
###############################################
from ultralytics import YOLO
import cv2
import numpy as np
import pyrealsense2 as rs
from opcua import Client, ua
import time


###############################################
# 2) OPC UA SETUP
###############################################
def opcua_connect(endpoint="opc.tcp://192.168.0.1:4840"):
    """Connect to OPC UA server and return client + nodes."""

    client = Client(endpoint)
    client.connect()
    print("[OPCUA] Connected")

    # TODO: Replace these placeholders with your real Node IDs
    node_x = client.get_node("ns=4;i=2")
    node_y = client.get_node("ns=4;i=3")
    node_z = client.get_node("ns=4;i=4")
    node_angle = client.get_node("ns=4;i=5")

    return client, node_x, node_y, node_z, node_angle


def opcua_write(node_x, node_y, node_z, node_angle, x, y, z, ang):
    """Write X/Y/Z/Angle continuously to the PLC."""
    try:
        node_x.set_value(ua.Variant(x, ua.VariantType.Float))
        node_y.set_value(ua.Variant(y, ua.VariantType.Float))
        node_z.set_value(ua.Variant(z, ua.VariantType.Float))
        node_angle.set_value(ua.Variant(ang, ua.VariantType.Float))

        print(f"[OPCUA] X={x:.1f}  Y={y:.1f}  Z={z:.1f}  A={ang:.1f}")
    except Exception as e:
        print("[OPCUA] Write error:", e)


###############################################
# 3) YOLO OBB DETECTION
###############################################
def run_detection(model, color_frame, depth_frame, intrinsics, img_w, img_h):
    """Run YOLO → return list of detections."""

    img = np.asanyarray(color_frame.get_data())
    results = model(img, conf=0.5, iou=0.5)[0]

    detections = []

    if results.obb is None:
        return detections, img

    obb = results.obb
    for i in range(len(obb.cls)):
        cls = int(obb.cls[i])
        conf = float(obb.conf[i])

        poly = obb.xyxyxyxy[i].cpu().numpy().astype(int)
        x, y, w, h, r = obb.xywhr[i].cpu().numpy().astype(float)

        # OBB center
        c = np.mean(poly, axis=0).astype(int)
        cx = int(np.clip(c[0], 0, img_w - 1))
        cy = int(np.clip(c[1], 0, img_h - 1))

        # Depth extraction
        d = depth_frame.get_distance(cx, cy)
        if d <= 0:
            continue

        # Pixel → 3D camera point
        p_cam = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], d)

        detections.append({
            "class": model.names[cls],
            "conf": conf,
            "poly": poly,
            "cx": cx, "cy": cy,
            "w": w, "h": h,
            "r_rad": float(r),
            "p_cam": p_cam
        })

    return detections, img


###############################################
# 4) DRAWING
###############################################
def draw(img, dets):
    """Draw OBB + labels."""
    out = img.copy()

    for d in dets:
        cx, cy = d["cx"], d["cy"]
        poly = d["poly"]
        cv2.polylines(out, [poly.reshape((-1, 1, 2))], True, (0, 255, 0), 2)

        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)

        cv2.putText(out,
                    f"{d['class']} {d['conf']:.2f}",
                    (cx - 50, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 200, 0), 1)
    return out


###############################################
# 5) TRANSFORMATION + MATH
###############################################
# Fixed calibration camera → gripper
R_gc = np.array([
    [0, -1, 0],
    [0,  0, 1],
    [-1, 0, 0]
], dtype=float)
t_gc = np.array([0.085, -0.220, 0.040], dtype=float)


def cam_to_gripper(p):
    """3D camera → 3D gripper."""
    p = np.array(p).reshape(3, 1)
    return (R_gc @ p + t_gc.reshape(3, 1)).flatten()


def major_axis_vector(r, w, h):
    """Unit vector of major OBB axis in camera frame."""
    if h > w:
        r += np.pi / 2
    return np.array([np.cos(r), np.sin(r), 0.0])


def vector_angle_xy(v):
    """Angle of 2D vector in XY plane."""
    return np.degrees(np.arctan2(v[1], v[0])) % 180


def compute_xyz_angle(det):
    """Compute XYZ + angle for one detection."""

    # XYZ in gripper frame
    p_cam = det["p_cam"]
    p_g = cam_to_gripper(p_cam)
    xg, yg, zg = p_g * 1000.0  # mm

    # Angle in gripper frame
    v_cam = major_axis_vector(det["r_rad"], det["w"], det["h"])
    v_grip = R_gc @ v_cam
    ang = vector_angle_xy(v_grip)

    return xg, yg, zg, ang


###############################################
# 6) MAIN LOOP
###############################################
def main():
    model = YOLO("epoch85.pt")

    # Camera init
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipeline.start(cfg)

    align = rs.align(rs.stream.color)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    img_w, img_h = intr.width, intr.height

    # OPC UA
    client, nX, nY, nZ, nA = opcua_connect()

    # Last known values (continuous streaming)
    lastX = lastY = lastZ = lastA = 0.0

    print("[SYSTEM] Running… Press ESC to quit.")

    try:
        while True:
            # Get frames
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            c = aligned.get_color_frame()
            d = aligned.get_depth_frame()
            if not c or not d:
                continue

            # Detection
            dets, img = run_detection(model, c, d, intr, img_w, img_h)

            # Update XYZ + angle if detection found
            if dets:
                x, y, z, a = compute_xyz_angle(dets[0])
                lastX, lastY, lastZ, lastA = x, y, z, a

            # Continuous OPCUA write
            opcua_write(nX, nY, nZ, nA, lastX, lastY, lastZ, lastA)

            # Draw + display
            vis = draw(img, dets)
            cv2.imshow("OBB + XYZ + Angle + OPCUA", vis)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(0.05)

    finally:
        client.disconnect()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[CLEANUP] Done.")


if __name__ == "__main__":
    main()

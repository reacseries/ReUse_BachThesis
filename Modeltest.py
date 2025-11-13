import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# -------------------------------
# 1. Initialize RealSense Camera
# -------------------------------
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

# -------------------------------
# 2. Load YOLOv11-OBB Model
# -------------------------------
model = YOLO("epoch85.pt")  # <<-- change to your model path
print("✅ YOLOv11 OBB model loaded")

# -------------------------------
# 3. Main Processing Loop
# -------------------------------
try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert to numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # -------------------------------
        # 4. YOLO Inference
        # -------------------------------
        results = model(color_image, task="obb", verbose=False)

        for r in results:
            boxes = r.obb
            if boxes is None:
                continue

            for box in boxes:
                # Oriented bounding box vertices (8 values)
                xyxyxyxy = box.xyxyxyxy[0].cpu().numpy().reshape(-1, 2).astype(int)

                # Get [x_center, y_center, width, height, rotation]
                xywhr = box.xywhr[0].cpu().numpy()
                x_center, y_center, w, h, angle = xywhr
                x_center, y_center = int(x_center), int(y_center)

                # Get depth at center
                z_m = depth_frame.get_distance(x_center, y_center)
                z_mm = z_m * 1000  # convert to mm

                # -------------------------------
                # 5. Draw Oriented Box + Overlay
                # -------------------------------
                # Draw the oriented polygon
                cv2.polylines(color_image, [xyxyxyxy], isClosed=True, color=(0, 255, 0), thickness=2)

                # Draw center point
                cv2.circle(color_image, (x_center, y_center), 4, (0, 0, 255), -1)

                # Place text slightly below the lowest vertex
                bottom_y = np.max(xyxyxyxy[:, 1])
                left_x = np.min(xyxyxyxy[:, 0])

                text = f"X:{x_center}  Y:{y_center}  Z:{z_mm:.1f}mm"
                cv2.putText(color_image, text, (left_x, bottom_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

                print(text)

        # Display
        cv2.imshow("YOLOv11-OBB (Oriented Bounding Boxes)", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

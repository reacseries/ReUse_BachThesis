import cv2
import numpy as np
import pyrealsense2 as rs
import os
from datetime import datetime

# ==== USER SETTINGS ====
CHECKERBOARD = (7,5)   # inner corners (cols, rows)
SQUARE_SIZE = 20.0      # mm
SAVE_DIR = "../captures"
# ========================

os.makedirs(SAVE_DIR, exist_ok=True)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

print("\n--- Hand-Eye Data Capture ---")
print("Press SPACE to capture (checkerboard must be visible).")
print("Press ESC to quit.\n")

i = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Try to find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FAST_CHECK
        )

        display = color_image.copy()
        if ret:
            # Refine corner accuracy
            corners_subpix = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners_subpix, ret)
            cv2.putText(display, "Checkerboard detected - Press SPACE to save",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No checkerboard detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("RealSense Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32 and ret:  # SPACE + checkerboard visible
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(SAVE_DIR, f"img_{i:03d}_{timestamp}.png")
            npy_path = os.path.join(SAVE_DIR, f"corners_{i:03d}_{timestamp}.npy")

            cv2.imwrite(img_path, color_image)
            np.save(npy_path, corners)

            print(f"[{i:02d}] Saved image + corners → {img_path}")
            print("→ Record robot pose for this image.\n")
            i += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Stopped RealSense pipeline.")

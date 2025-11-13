import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# ---- Settings ----
SAVE_DEPTH = False   # set to False if you only want color images
COLOR_RES = (640, 480)  # (width, height)
DEPTH_RES = (640, 480)
FPS = 30

# Output directories (as requested: "model data")
BASE_DIR = Path("../model data")
COLOR_DIR = BASE_DIR / "color2"
DEPTH_DIR = BASE_DIR / "depth"

def ensure_dirs():
    COLOR_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_DEPTH:
        DEPTH_DIR.mkdir(parents=True, exist_ok=True)

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def main():
    ensure_dirs()

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_RES[0], COLOR_RES[1], rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_RES[0], DEPTH_RES[1], rs.format.z16, FPS)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print("Failed to start RealSense pipeline. Check USB connection and permissions.\n", e)
        sys.exit(1)

    # Align depth to color so saved depth maps match the color frame
    align = rs.align(rs.stream.color)

    # For nicer on-screen visualization of depth (we'll still save the raw 16-bit depth)
    colorizer = rs.colorizer()

    print("Controls: [SPACE] = save frame(s), [q]/[ESC] = quit.")
    print(f"Saves to: {COLOR_DIR} {'and ' + str(DEPTH_DIR) if SAVE_DEPTH else ''}")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())  # 16-bit depth in millimeters

            # On-screen depth (colorized for visualization only)
            depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # Side-by-side preview
            vis = np.hstack((color_image, depth_vis))
            cv2.imshow("D435i - Color (left) | Depth (right)  [SPACE=save, q/ESC=quit]", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                # ESC or 'q'
                break
            if key == 32:  # SPACE
                t = timestamp()
                color_path = COLOR_DIR / f"color_{t}.png"
                cv2.imwrite(str(color_path), color_image)

                if SAVE_DEPTH:
                    # Save depth as 16-bit PNG (millimeters)
                    depth_path = DEPTH_DIR / f"depth_{t}.png"
                    # Ensure proper 16-bit write
                    cv2.imwrite(str(depth_path), depth_image)

                print(f"Saved: {color_path}" + (f" | {depth_path}" if SAVE_DEPTH else ""))

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

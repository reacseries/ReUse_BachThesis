import cv2
import numpy as np
from ultralytics import YOLO

# Load your trained OBB model
model = YOLO("latestmodel.pt")

def draw_line_and_angle(image, obb_box):
    # Extract OBB parameters (x, y, w, h, r) from YOLOv8/11-OBB
    x, y, w, h, r = obb_box.xywhr[0].cpu().numpy()

    # Determine the longer side (major axis)
    length = max(w, h)
    half_len = length / 2

    # Compute end points of the line through the center
    x1 = x - half_len * np.cos(r)
    y1 = y - half_len * np.sin(r)
    x2 = x + half_len * np.cos(r)
    y2 = y + half_len * np.sin(r)

    # Draw the OBB polygon
    pts = obb_box.xyxyxyxy[0].cpu().numpy().astype(int)
    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw the major-axis line through center
    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)

    # Compute the actual direction vector of the longer axis
    dx = x2 - x1
    dy = y2 - y1

    # Compute angle relative to the image's x-axis (horizontal base)
    # OpenCV image coords: x→right, y→down, so we invert dy for geometric correctness
    angle_rad = np.arctan2(-dy, dx)
    angle_deg = np.degrees(angle_rad)

    # Normalize angle to [0, 180)
    if angle_deg < 0:
        angle_deg += 180

    # Display angle text near the box
    text = f"{angle_rad:.2f} rad / {angle_deg:.1f}°"
    cv2.putText(image, text, (int(x) - 60, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return angle_rad, angle_deg


# Live camera or video feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        if hasattr(r, 'obb') and r.obb is not None:
            for obb_box in r.obb:
                angle_rad, angle_deg = draw_line_and_angle(frame, obb_box)
                print(f"Angle relative to base: {angle_rad:.3f} rad ({angle_deg:.2f}°)")

    cv2.imshow("OBB Detection with Base Angle", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

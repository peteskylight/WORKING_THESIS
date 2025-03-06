import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (in degrees):
    a (x1, y1), b (x2, y2), c (x3, y3)
    """
    if any(p is None for p in [a, b, c]):
        return None
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def detect_head_movements(keypoints):
    """
    Detect head movement based on YOLOv8 keypoints.
    keypoints: Dictionary with keypoint names as keys and (x, y) tuples as values.
    """
    def get_keypoint(name):
        return keypoints.get(name, None)
    
    nose = get_keypoint("nose")
    left_eye = get_keypoint("left_eye")
    right_eye = get_keypoint("right_eye")
    left_ear = get_keypoint("left_ear")
    right_ear = get_keypoint("right_ear")
    left_shoulder = get_keypoint("left_shoulder")
    right_shoulder = get_keypoint("right_shoulder")
    
    if left_shoulder and right_shoulder:
        neck = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
    else:
        neck = None
    
    # Compute head tilt (lateral inclination)
    ear_line_angle = calculate_angle(left_ear, nose, right_ear)
    if ear_line_angle is not None and ear_line_angle > 15:
        head_tilt = "Tilting Right" if nose and left_ear and nose[1] < left_ear[1] else "Tilting Left"
    else:
        head_tilt = "Neutral"
    
    # Compute head turn (yaw rotation)
    if nose and right_eye and nose[0] > right_eye[0]:
        head_turn = "Turning Left"
    elif nose and left_eye and nose[0] < left_eye[0]:
        head_turn = "Turning Right"
    else:
        head_turn = "Facing Forward"
    
    # Compute head nodding (up-down movement)
    if nose and neck:
        nod_angle = calculate_angle(nose, neck, (neck[0], neck[1] - 1))  # Compare with vertical
        if nod_angle is not None:
            if nod_angle > 20:
                head_nod = "Looking Down"
            elif nod_angle < -20:
                head_nod = "Looking Up"
            else:
                head_nod = "Neutral"
        else:
            head_nod = "Unknown"
    else:
        head_nod = "Unknown"
    
    return {"head_tilt": head_tilt, "head_turn": head_turn, "head_nod": head_nod}

# Load YOLOv8 Pose Model
model = YOLO("yolov8n-pose.pt")

# Video processing
cap = cv2.VideoCapture(r"C:\Users\peter\Desktop\WORKING THESIS FILES\RESOURCES\ACTUAL SURVEY\MARCH 1\TestIndiv2.mp4")
output_video = r"C:\Users\peter\Desktop\WORKING THESIS FILES\MODULAR_TESTING\Tests\HEAD POSTURE\outputs\HeadPosture1.mp4"
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

time_series = []
tilt_series = []
turn_series = []
nod_series = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame is None:
        print(f"Frame {frame_count} could not be read. Exiting.")
        break

    # Run YOLOv8 Pose Estimation
    results = model(frame)
    if results and results[0].keypoints is not None:
        for kp in results[0].keypoints.xy.cpu().numpy():
            keypoints_dict = {name: tuple(kp[i]) if i < len(kp) else None for i, name in enumerate([
                "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder"])
            }
            result = detect_head_movements(keypoints_dict)
            
            # Draw keypoints
            for point in keypoints_dict.values():
                if point is not None:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
            
            # Draw results on frame
            cv2.putText(frame, result["head_tilt"], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, result["head_turn"], (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, result["head_nod"], (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            time_series.append(frame_count)
            tilt_series.append(result["head_tilt"])
            turn_series.append(result["head_turn"])
            nod_series.append(result["head_nod"])
    
    out.write(frame)
    frame_count += 1

cap.release()
out.release()

# Ensure output directory exists
plt.savefig(r"C:\Users\peter\Desktop\WORKING THESIS FILES\MODULAR_TESTING\Tests\HEAD POSTURE\outputs\movement_graph.png")
plt.show()

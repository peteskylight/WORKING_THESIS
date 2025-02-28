import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ultralytics import YOLO

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt")

# Define keypoint indices (COCO format)
EAR_R_IDX, EAR_L_IDX = 4, 3
SHOULDER_R_IDX, SHOULDER_L_IDX = 6, 5
WRIST_R_IDX, WRIST_L_IDX = 10, 9
HIP_R_IDX, HIP_L_IDX = 12, 11

# Thresholds (Adjust based on testing)
ANGLE_RAISE_THRESHOLD = 60  # Angle threshold for raising arm
ANGLE_SIDE_THRESHOLD = 40   # Angle threshold for sidewards extension
ANGLE_FORWARD_THRESHOLD = 20  # Angle threshold for forward/backward extension

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_angle(a, b, c):
    ab = np.array(a) - np.array(b)
    cb = np.array(c) - np.array(b)
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def process_video(video_path, output_video, graph_pdf):
    cap = cv2.VideoCapture(video_path)
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    FRAME_WIDTH = int(cap.get(3))
    FRAME_HEIGHT = int(cap.get(4))
    FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, FPS, FRAME_SIZE)
    if not out.isOpened():
        print("Error: Could not open VideoWriter")
        return

    timestamps = []
    right_arm_distances = []
    left_arm_distances = []
    right_arm_postures = []
    left_arm_postures = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy()
            if keypoints.shape[0] == 0:
                continue

            if keypoints.shape[1] < max(WRIST_R_IDX, WRIST_L_IDX, HIP_R_IDX, HIP_L_IDX) + 1:
                continue

            right_shoulder = keypoints[0, SHOULDER_R_IDX]
            right_wrist = keypoints[0, WRIST_R_IDX]
            right_hip = keypoints[0, HIP_R_IDX]
            left_shoulder = keypoints[0, SHOULDER_L_IDX]
            left_wrist = keypoints[0, WRIST_L_IDX]
            left_hip = keypoints[0, HIP_L_IDX]

            right_dist = euclidean_distance(right_shoulder, right_wrist)
            left_dist = euclidean_distance(left_shoulder, left_wrist)
            right_angle = calculate_angle(right_hip, right_shoulder, right_wrist)
            left_angle = calculate_angle(left_hip, left_shoulder, left_wrist)

            right_posture = "Resting"
            if right_angle > ANGLE_RAISE_THRESHOLD:
                right_posture = "Raising Hand"
            elif right_angle > ANGLE_SIDE_THRESHOLD:
                right_posture = "Extending Sidewards"
            elif right_angle > ANGLE_FORWARD_THRESHOLD:
                right_posture = "Extending Forward"
            
            left_posture = "Resting"
            if left_angle > ANGLE_RAISE_THRESHOLD:
                left_posture = "Raising Hand"
            elif left_angle > ANGLE_SIDE_THRESHOLD:
                left_posture = "Extending Sidewards"
            elif left_angle > ANGLE_FORWARD_THRESHOLD:
                left_posture = "Extending Forward"

            timestamps.append(frame_idx / FPS)
            right_arm_distances.append(right_dist)
            left_arm_distances.append(left_dist)
            right_arm_postures.append(right_posture)
            left_arm_postures.append(left_posture)

            for keypoint in keypoints[0]:
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            action_text = f"Right: {right_posture}, Left: {left_posture}"
            cv2.putText(frame, action_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            out.write(frame)
        
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Processed video saved as {output_video}")
    plot_graph(timestamps, right_arm_distances, left_arm_distances, right_arm_postures, left_arm_postures, graph_pdf)

def plot_graph(timestamps, right_arm_distances, left_arm_distances, right_arm_postures, left_arm_postures, output_pdf):
    posture_mapping = {"Resting": 25, "Raising Hand": 75, "Extending Sidewards": 50, "Extending Forward": 100, "Extending Backward": 0}
    right_posture_values = [posture_mapping[p] for p in right_arm_postures]
    left_posture_values = [posture_mapping[p] for p in left_arm_postures]

    with PdfPages(output_pdf) as pdf:
        plt.figure(figsize=(11, 8.5))
        plt.plot(timestamps, right_arm_distances, label="Right Arm Distance", color="blue")
        plt.plot(timestamps, left_arm_distances, label="Left Arm Distance", color="red")
        plt.axhspan(0, 25, facecolor='gray', alpha=0.3, label="Resting")
        plt.axhspan(25, 50, facecolor='orange', alpha=0.3, label="Extending Sidewards")
        plt.axhspan(50, 75, facecolor='purple', alpha=0.3, label="Raising Hand")
        plt.axhspan(75, 100, facecolor='green', alpha=0.3, label="Extending Forward")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Distance")
        plt.title("Arm Movement Over Time")
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()
    print(f"Graph saved as {output_pdf}")

process_video(r"C:\Users\USER\Desktop\WORKING_THESIS\RESOURCES\Sample Vids\SampleVid3.mp4", "processed_video3.mp4", "movement_graph3.pdf")

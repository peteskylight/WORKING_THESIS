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
RAISE_Y_THRESHOLD = 40
SIDE_X_THRESHOLD = 60
REST_Y_THRESHOLD = 20
WAIST_Y_THRESHOLD = 30
STAND_SIT_THRESHOLD = 50

# Open video file
video_path = r"C:\Users\USER\Desktop\WORKING_THESIS\RESOURCES\Sample Vids\SampleVid1.mp4"
cap = cv2.VideoCapture(video_path)
FPS = cap.get(cv2.CAP_PROP_FPS)
FRAME_INTERVAL = int(FPS * 10)  # Capture frame every 10 seconds

# Store data for plotting
timestamps = []
right_arm_distances = []
left_arm_distances = []
right_arm_postures = []
left_arm_postures = []
frame_images = []  # Store extracted frames with keypoints
frame_times = []   # Time corresponding to extracted frames

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break when video ends

    results = model(frame)
    
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()

        if keypoints.shape[0] == 0:  # No detections
            continue

        # Extract keypoints
        right_ear = keypoints[0, EAR_R_IDX]
        right_shoulder = keypoints[0, SHOULDER_R_IDX]
        right_wrist = keypoints[0, WRIST_R_IDX]
        right_hip = keypoints[0, HIP_R_IDX]

        left_ear = keypoints[0, EAR_L_IDX]
        left_shoulder = keypoints[0, SHOULDER_L_IDX]
        left_wrist = keypoints[0, WRIST_L_IDX]
        left_hip = keypoints[0, HIP_L_IDX]

        # Time calculation
        time_sec = frame_idx / FPS

        # Compute distances
        right_y_dist = abs(right_ear[1] - right_wrist[1])
        right_x_dist = abs(right_shoulder[0] - right_wrist[0])
        right_wrist_to_waist = abs(right_wrist[1] - right_hip[1])

        left_y_dist = abs(left_ear[1] - left_wrist[1])
        left_x_dist = abs(left_shoulder[0] - left_wrist[0])
        left_wrist_to_waist = abs(left_wrist[1] - left_hip[1])

        # Determine arm posture (Right Arm)
        if right_y_dist > RAISE_Y_THRESHOLD:
            right_posture = "Raising Arm"
        elif right_x_dist > SIDE_X_THRESHOLD:
            right_posture = "Extending Sidewards"
        elif right_wrist_to_waist < WAIST_Y_THRESHOLD:
            right_posture = "Resting"
        else:
            right_posture = "Unknown"

        # Determine arm posture (Left Arm)
        if left_y_dist > RAISE_Y_THRESHOLD:
            left_posture = "Raising Arm"
        elif left_x_dist > SIDE_X_THRESHOLD:
            left_posture = "Extending Sidewards"
        elif left_wrist_to_waist < WAIST_Y_THRESHOLD:
            left_posture = "Resting"
        else:
            left_posture = "Unknown"

        # Store values
        timestamps.append(time_sec)
        right_arm_distances.append(right_y_dist)
        left_arm_distances.append(left_y_dist)
        right_arm_postures.append(right_posture)
        left_arm_postures.append(left_posture)

        # Save frame with keypoints every 10 seconds
        if frame_idx % FRAME_INTERVAL == 0:
            for keypoint in keypoints[0]:
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            frame_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_times.append(time_sec)

    frame_idx += 1

cap.release()

# Convert posture labels to numeric values for visualization
posture_mapping = {"Resting": 20, "Raising Arm": 60, "Extending Sidewards": 40, "Unknown": 0}

right_posture_values = [posture_mapping[p] for p in right_arm_postures]
left_posture_values = [posture_mapping[p] for p in left_arm_postures]

# Create PDF
pdf_path = "arm_movement_analysis.pdf"
with PdfPages(pdf_path) as pdf:
    # Determine number of pages if graph is too wide
    max_time_per_page = 30  # 30 seconds per page
    total_time = timestamps[-1] if timestamps else 0
    num_pages = max(1, int(np.ceil(total_time / max_time_per_page)))

    for i in range(num_pages):
        plt.figure(figsize=(11, 8.5))  # Landscape mode

        # Time range for this page
        start_time = i * max_time_per_page
        end_time = (i + 1) * max_time_per_page

        # Filter data for this page
        indices = [idx for idx, t in enumerate(timestamps) if start_time <= t < end_time]
        times_page = [timestamps[idx] for idx in indices]
        right_distances_page = [right_arm_distances[idx] for idx in indices]
        left_distances_page = [left_arm_distances[idx] for idx in indices]
        right_postures_page = [right_posture_values[idx] for idx in indices]

        # Plot line graphs
        plt.plot(times_page, right_distances_page, label="Right Arm Distance", color="blue")
        plt.plot(times_page, left_distances_page, label="Left Arm Distance", color="red")

        # Background shading for posture indication
        plt.fill_between(times_page, 0, 100, where=np.array(right_postures_page) == 20, color='green', alpha=0.2, label="Resting")
        plt.fill_between(times_page, 0, 100, where=np.array(right_postures_page) == 40, color='orange', alpha=0.2, label="Extending Sidewards")
        plt.fill_between(times_page, 0, 100, where=np.array(right_postures_page) == 60, color='purple', alpha=0.2, label="Raising Arm")

        # Labels and legend
        plt.xlabel("Time (seconds)")
        plt.ylabel("Distance")
        plt.title(f"Arm Movement Over Time ({start_time}-{end_time} sec)")
        plt.legend()
        plt.grid(True)

        # Add graph to PDF
        pdf.savefig()
        plt.close()

        # Add corresponding frame images every 10 seconds
        for idx, img in enumerate(frame_images):
            if start_time <= frame_times[idx] < end_time:
                fig, ax = plt.subplots(figsize=(11, 6))
                ax.imshow(img)
                ax.set_title(f"Frame at {frame_times[idx]:.1f} sec")
                ax.axis("off")
                pdf.savefig(fig)
                plt.close()

print(f"Graph saved as {pdf_path}")

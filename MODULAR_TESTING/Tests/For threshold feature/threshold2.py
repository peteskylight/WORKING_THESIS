import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt")

# Define keypoint indices (COCO format)
EAR_R_IDX, EAR_L_IDX = 4, 3  # Right and Left ear
SHOULDER_R_IDX, SHOULDER_L_IDX = 6, 5
ELBOW_R_IDX, ELBOW_L_IDX = 8, 7
WRIST_R_IDX, WRIST_L_IDX = 10, 9
HIP_R_IDX, HIP_L_IDX = 12, 11  # Right and Left hip

# Thresholds (Adjust based on testing)
RAISE_Y_THRESHOLD = 40  # Arm tip above ear
SIDE_X_THRESHOLD = 60   # Arm extended sideward
REST_Y_THRESHOLD = 20   # Arm near shoulder height
WAIST_Y_THRESHOLD = 30  # Wrist near waist = Resting
STAND_SIT_THRESHOLD = 50  # Hip height to determine sitting or standing

# Open video file
video_path = r"C:\Users\USER\Desktop\WORKING_THESIS\RESOURCES\Sample Vids\SampleVid1.mp4"
cap = cv2.VideoCapture(video_path)

frame_idx = 0
postures = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break when video ends

    results = model(frame)
    
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()

        if keypoints.shape[0] == 0:  # No detections
            continue

        # Extract keypoints for right and left arms
        right_ear = keypoints[0, EAR_R_IDX]
        right_shoulder = keypoints[0, SHOULDER_R_IDX]
        right_wrist = keypoints[0, WRIST_R_IDX]
        right_hip = keypoints[0, HIP_R_IDX]

        left_ear = keypoints[0, EAR_L_IDX]
        left_shoulder = keypoints[0, SHOULDER_L_IDX]
        left_wrist = keypoints[0, WRIST_L_IDX]
        left_hip = keypoints[0, HIP_L_IDX]

        # Determine if person is sitting or standing (based on hip height)
        person_height = abs(right_ear[1] - right_hip[1])
        is_sitting = person_height < STAND_SIT_THRESHOLD

        # Compute distances
        right_y_dist = right_ear[1] - right_wrist[1]  # Y-axis (vertical)
        right_x_dist = abs(right_shoulder[0] - right_wrist[0])  # X-axis (horizontal)
        right_wrist_to_waist = abs(right_wrist[1] - right_hip[1])  # Wrist to waist

        left_y_dist = left_ear[1] - left_wrist[1]
        left_x_dist = abs(left_shoulder[0] - left_wrist[0])
        left_wrist_to_waist = abs(left_wrist[1] - left_hip[1])

        # Classify posture for right arm
        if right_y_dist > RAISE_Y_THRESHOLD:
            right_posture = "Raising Arm"
        elif right_x_dist > SIDE_X_THRESHOLD:
            right_posture = "Extending Sidewards"
        elif right_wrist_to_waist < WAIST_Y_THRESHOLD:
            if is_sitting:
                right_posture = "Resting on Table"
            else:
                right_posture = "Resting Downwards"
        else:
            right_posture = "Unknown"

        # Classify posture for left arm
        if left_y_dist > RAISE_Y_THRESHOLD:
            left_posture = "Raising Arm"
        elif left_x_dist > SIDE_X_THRESHOLD:
            left_posture = "Extending Sidewards"
        elif left_wrist_to_waist < WAIST_Y_THRESHOLD:
            if is_sitting:
                left_posture = "Resting on Table"
            else:
                left_posture = "Resting Downwards"
        else:
            left_posture = "Unknown"

        # Store results
        postures.append((frame_idx, right_posture, left_posture))

        # Display text on video
        cv2.putText(frame, f"Right Arm: {right_posture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Left Arm: {left_posture}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw keypoints
        for keypoint in keypoints[0]:
            x, y = int(keypoint[0]), int(keypoint[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Pose Estimation", frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Plot arm postures over time
plt.figure(figsize=(10, 5))
frames, right_arms, left_arms = zip(*postures)
plt.plot(frames, [p.replace("Raising Arm", "1").replace("Resting Downwards", "0").replace("Resting on Table", "0.5").replace("Extending Sidewards", "2") for p in right_arms], label="Right Arm")
plt.plot(frames, [p.replace("Raising Arm", "1").replace("Resting Downwards", "0").replace("Resting on Table", "0.5").replace("Extending Sidewards", "2") for p in left_arms], label="Left Arm")
plt.xlabel("Frame")
plt.ylabel("Posture State (0=Rest Down, 0.5=Rest Table, 1=Raise, 2=Side)")
plt.legend()
plt.title("Arm Posture Over Time")
plt.show()

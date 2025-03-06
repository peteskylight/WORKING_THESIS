import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ultralytics import YOLO

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt")  # Use a pose estimation model

# Keypoint pairs for drawing skeleton lines
SKELETON_CONNECTIONS = [(5, 7), (7, 9),  # Left arm
                        (6, 8), (8, 10)]  # Right arm

# Movement categories with numerical values for graphing
MOVEMENT_CATEGORIES = {
    "Resting Downward": 1,
    "Extending Forward": 2,
    "Extending Sidewards Right": 3,  # Person's right arm
    "Extending Sidewards Left": 4,   # Person's left arm
    "Extending Backward": 5,
    "Unknown": 0
}

def calculate_angle(p1, p2, p3):
    """Calculate the elbow joint angle between shoulder, elbow, and wrist."""
    v1 = np.array(p1) - np.array(p2)  # Shoulder to elbow
    v2 = np.array(p3) - np.array(p2)  # Wrist to elbow
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)  # Convert to degrees

def classify_arm_movement(shoulder, elbow, wrist, is_left_arm):
    """Classify arm movement with mirroring correction for camera perspective."""
    if any(k is None or np.isnan(k).any() for k in [shoulder, elbow, wrist]):
        return "Unknown"

    angle = calculate_angle(shoulder, elbow, wrist)
    shoulder_wrist_distance = np.linalg.norm(np.array(shoulder) - np.array(wrist))
    shoulder_elbow_distance = np.linalg.norm(np.array(shoulder) - np.array(elbow))
    movement_score = shoulder_wrist_distance / shoulder_elbow_distance

    if angle < 45 and movement_score < 0.6:
        return "Resting Downward"
    elif 45 <= angle < 90 and 0.6 <= movement_score < 1.0:
        return "Extending Forward"
    elif 90 <= angle < 135 and 0.6 <= movement_score < 1.0:
        return "Extending Sidewards Right" if is_left_arm else "Extending Sidewards Left"
    elif angle >= 135 and movement_score >= 1.0:
        return "Extending Backward"
    return "Unknown"

def draw_skeleton_and_angles(frame, keypoints):
    """Draw pose skeleton, keypoints, and annotate joint angles on the frame."""
    for person in keypoints:
        # Draw keypoints
        for i, (x, y) in enumerate(person):
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)  # Yellow dots

        # Draw skeleton lines
        for (p1, p2) in SKELETON_CONNECTIONS:
            x1, y1 = map(int, person[p1])
            x2, y2 = map(int, person[p2])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue lines

        # Compute and annotate angles
        for (shoulder_idx, elbow_idx, wrist_idx) in [(5, 7, 9), (6, 8, 10)]:  # Left and Right arm
            angle = calculate_angle(person[shoulder_idx], person[elbow_idx], person[wrist_idx])
            ex, ey = map(int, person[elbow_idx])
            cv2.putText(frame, f"{int(angle)}°", (ex - 20, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def process_video(input_video, output_video):
    """Process video to detect and classify arm movements."""
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = 0
    time_series = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # YOLO Pose Estimation
        keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []

        draw_skeleton_and_angles(frame, keypoints)  # Draw keypoints and angles

        for person in keypoints:
            right_shoulder, right_elbow, right_wrist = person[5], person[7], person[9]
            left_shoulder, left_elbow, left_wrist = person[6], person[8], person[10]

            left_movement = classify_arm_movement(left_shoulder, left_elbow, left_wrist, is_left_arm=True)
            right_movement = classify_arm_movement(right_shoulder, right_elbow, right_wrist, is_left_arm=False)

            # Store data for visualization
            time_series.append((frame_count / fps, left_movement, right_movement))

        # Move labels to the right side
        cv2.putText(frame, f"Left Arm: {left_movement}", (width - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Right Arm: {right_movement}", (width - 200, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    generate_graph(time_series,
                   r"C:\Users\peter\Desktop\WORKING THESIS FILES\MODULAR_TESTING\Tests\Thresholding 2\outputs\movements.pdf")



def generate_graph(time_series, pdf_filename):
    """Generate and save a time-based event graph with movement sectioning."""
    times = [t[0] for t in time_series]
    left_movements = [t[1] for t in time_series]
    right_movements = [t[2] for t in time_series]

    for i in range(1, len(left_movements)):
        if left_movements[i] == "Unknown":
            left_movements[i] = left_movements[i - 1]
        if right_movements[i] == "Unknown":
            right_movements[i] = right_movements[i - 1]

    left_values = [MOVEMENT_CATEGORIES[m] for m in left_movements]
    right_values = [MOVEMENT_CATEGORIES[m] for m in right_movements]

    plt.figure(figsize=(10, 5))
    plt.plot(times, left_values, marker='o', linestyle='-', color='b', label="Left Arm")
    plt.plot(times, right_values, marker='x', linestyle='-', color='r', label="Right Arm")

    plt.yticks(list(MOVEMENT_CATEGORIES.values()), MOVEMENT_CATEGORIES.keys())
    plt.xlabel("Time (seconds)")
    plt.ylabel("Arm Movement")
    plt.title("Arm Movement Over Time")
    plt.legend()
    plt.grid()

    with PdfPages(pdf_filename) as pdf:
        pdf.savefig()
        plt.close()

    print(f"Graph saved as {pdf_filename}")


# Run the program
process_video(r"C:\Users\peter\Desktop\WORKING THESIS FILES\RESOURCES\ACTUAL SURVEY\MARCH 1\TestIndiv2.mp4",
              r"C:\Users\peter\Desktop\WORKING THESIS FILES\MODULAR_TESTING\Tests\Thresholding 2\outputs\processed_video3.mp4")

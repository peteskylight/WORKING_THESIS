import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ultralytics import YOLO

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt")

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6)]  # Left & Right arms + shoulders

# Movement categories for graphing
MOVEMENT_CATEGORIES = {
    "Resting Downward": 1,
    "Extending Forward": 2,
    "Extending Sidewards Right": 3,
    "Extending Sidewards Left": 4,
    "Extending Backward": 5,
    "Unknown": 0
}

def calculate_angle(p1, p2, p3):
    """Calculate the angle formed by three points (in degrees)."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def classify_arm_movement(shoulder, elbow, wrist, opp_shoulder, is_left_arm):
    """Classify arm movement based on new angle definitions."""
    if any(k is None or np.isnan(k).any() for k in [shoulder, elbow, wrist, opp_shoulder]):
        return "Unknown"

    # Calculate angles
    elbow_wrist_angle = calculate_angle(elbow, wrist, shoulder)
    shoulder_shoulder_vector = np.array(shoulder) - np.array(opp_shoulder)
    shoulder_elbow_vector = np.array(elbow) - np.array(shoulder)
    
    # Angle between shoulder-shoulder and shoulder-elbow
    shoulder_elbow_angle = calculate_angle(opp_shoulder, shoulder, elbow)
    
    # Check if arm is extending
    if 120 <= elbow_wrist_angle <= 180 and 120 <= shoulder_elbow_angle <= 225:
        return "Extending Sidewards Right" if is_left_arm else "Extending Sidewards Left"
    return "Resting Downward"

def draw_skeleton_and_angles(frame, keypoints):
    """Draw keypoints, skeleton lines, and annotate angles on joints."""
    for person in keypoints:
        for i, (x, y) in enumerate(person):
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)  # Yellow keypoints

        for (p1, p2) in SKELETON_CONNECTIONS:
            x1, y1 = map(int, person[p1])
            x2, y2 = map(int, person[p2])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Annotate angles
        for (shoulder_idx, elbow_idx, wrist_idx, opp_shoulder_idx) in [(5, 7, 9, 6), (6, 8, 10, 5)]:
            elbow_angle = calculate_angle(person[shoulder_idx], person[elbow_idx], person[wrist_idx])
            shoulder_angle = calculate_angle(person[opp_shoulder_idx], person[shoulder_idx], person[elbow_idx])
            ex, ey = map(int, person[elbow_idx])
            sx, sy = map(int, person[shoulder_idx])
            cv2.putText(frame, f"{int(elbow_angle)}°", (ex - 20, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"{int(shoulder_angle)}°", (sx - 20, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def process_video(input_video, output_video):
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

        results = model(frame)
        keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []

        draw_skeleton_and_angles(frame, keypoints)

        for person in keypoints:
            right_shoulder, right_elbow, right_wrist = person[5], person[7], person[9]
            left_shoulder, left_elbow, left_wrist = person[6], person[8], person[10]
            shoulder_center = (person[5] + person[6]) / 2

            left_movement = classify_arm_movement(left_shoulder, left_elbow, left_wrist, right_shoulder, is_left_arm=True)
            right_movement = classify_arm_movement(right_shoulder, right_elbow, right_wrist, left_shoulder, is_left_arm=False)

            time_series.append((frame_count / fps, left_movement, right_movement))

        # Move labels to the upper left
        cv2.putText(frame, f"Left Arm: {left_movement}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Right Arm: {right_movement}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
process_video(r"C:\Users\peter\Desktop\WORKING THESIS FILES\MODULAR_TESTING\Tests\Thresholding 2\Validation.mp4",
              r"C:\Users\peter\Desktop\WORKING THESIS FILES\MODULAR_TESTING\Tests\Thresholding 2\outputs\processed_video3.mp4")


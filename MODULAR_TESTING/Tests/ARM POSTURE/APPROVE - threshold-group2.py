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

# Adjustable movement thresholds
ARM_POSTURE_THRESHOLDS = {
    "elbow_wrist": (120, 180),  # Acceptable range for elbow-wrist angle
    "shoulder_elbow": (120, 225)  # Acceptable range for shoulder-elbow angle
}

def calculate_angle(p1, p2, p3):
    """Calculate the angle formed by three points (in degrees)."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def classify_arm_movement(elbow_wrist_angle, shoulder_elbow_angle, is_left_arm):
    """Classify arm movement based on defined thresholds."""
    if (ARM_POSTURE_THRESHOLDS["elbow_wrist"][0] <= elbow_wrist_angle <= ARM_POSTURE_THRESHOLDS["elbow_wrist"][1] and
        ARM_POSTURE_THRESHOLDS["shoulder_elbow"][0] <= shoulder_elbow_angle <= ARM_POSTURE_THRESHOLDS["shoulder_elbow"][1]):
        return "Extending Sidewards Right" if is_left_arm else "Extending Sidewards Left"
    return "Resting Downward"

def draw_skeleton_and_angles(frame, keypoints, angle_data):
    """Draw keypoints, skeleton lines, and annotate angles on joints for multiple persons."""
    for person_id, person in enumerate(keypoints):
        for i, (x, y) in enumerate(person):
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)  # Yellow keypoints

        for (p1, p2) in SKELETON_CONNECTIONS:
            x1, y1 = map(int, person[p1])
            x2, y2 = map(int, person[p2])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Calculate and annotate angles
        right_elbow_wrist = calculate_angle(person[5], person[7], person[9])
        right_shoulder_elbow = calculate_angle(person[6], person[5], person[7])
        left_elbow_wrist = calculate_angle(person[6], person[8], person[10])
        left_shoulder_elbow = calculate_angle(person[5], person[6], person[8])
        
        angles = {
            "Right Elbow-Wrist": right_elbow_wrist,
            "Right Shoulder-Elbow": right_shoulder_elbow,
            "Left Elbow-Wrist": left_elbow_wrist,
            "Left Shoulder-Elbow": left_shoulder_elbow,
        }
        angle_data.append((person_id, angles))
        
        # Display angles on frame
        cv2.putText(frame, f"{int(right_elbow_wrist)}°", (int(person[7][0]), int(person[7][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"{int(right_shoulder_elbow)}°", (int(person[5][0]), int(person[5][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def process_video(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = 0
    angle_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []

        draw_skeleton_and_angles(frame, keypoints, angle_data)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    export_angles_to_pdf(angle_data, r"C:\Users\peter\Desktop\WORKING THESIS FILES\MODULAR_TESTING\Tests\Thresholding 2\outputs\angles_report-GOOD.pdf")


def export_angles_to_pdf(angle_data, filename):
    """Export time-based joint angle graphs for multiple persons stacked vertically."""
    with PdfPages(filename) as pdf:
        times = list(range(len(angle_data)))
        people_ids = sorted(set(person_id for person_id, _ in angle_data))
        num_plots = (len(people_ids) + 9) // 10
        
        fig, axes = plt.subplots(nrows=min(10, len(people_ids)), ncols=1, figsize=(10, 5 * min(10, len(people_ids))), sharex=True)
        if len(people_ids) == 1:
            axes = [axes]
        
        for plot_index, person_id in enumerate(people_ids[:10]):
            right_angles = [data[1].get("Right Elbow-Wrist", 0) for data in angle_data if data[0] == person_id]
            left_angles = [data[1].get("Left Elbow-Wrist", 0) for data in angle_data if data[0] == person_id]
            
            axes[plot_index].plot(times[:len(right_angles)], right_angles, label=f"Person {person_id} Right Arm", color='blue')
            axes[plot_index].plot(times[:len(left_angles)], left_angles, label=f"Person {person_id} Left Arm", linestyle='dashed', color='red')
            
            axes[plot_index].set_ylabel("Angle (degrees)")
            axes[plot_index].set_ylim(0, 360)
            axes[plot_index].legend()
            axes[plot_index].grid()
        
        axes[-1].set_xlabel("Time (seconds)")
        fig.suptitle("Arm Angles Over Time (Stacked View)")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    print(f"Angles exported to {filename}")


process_video(r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\ACTUAL SURVEY\MARCH 1\test.mp4",
              r"C:\Users\Bennett\Documents\WORKING_THESIS\MODULAR_TESTING\Tests\ARM POSTURE\outputs\CHECKING.mp4")


import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ultralytics import YOLO

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt")

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6)]  # Left & Right arms + shoulders

# Posture classification criteria
def classify_posture(shoulder_angle, elbow_angle):
    if shoulder_angle >= 150 and elbow_angle <= 45:
        return "Raising Arm"
    elif shoulder_angle >= 150 and elbow_angle < 180:
        return "Raising Hand"
    elif shoulder_angle >= 130 and elbow_angle >= 160:
        return "Extending Sidewards"
    elif shoulder_angle < 120 and elbow_angle < 180:
        return "Resting Arm"
    return "Unknown"

def calculate_angle(p1, p2, p3):
    """Calculate the angle formed by three points (in degrees)."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def draw_skeleton_and_angles(frame, keypoints, angle_data):
    """Draw keypoints, skeleton lines, annotate angles, and classify posture."""
    for person_id, person in enumerate(keypoints):
        bbox_x_min = min([kp[0] for kp in person])
        bbox_y_min = min([kp[1] for kp in person])
        bbox_x_max = max([kp[0] for kp in person])
        bbox_y_max = max([kp[1] for kp in person])

        cv2.rectangle(frame, (int(bbox_x_min), int(bbox_y_min)), (int(bbox_x_max), int(bbox_y_max)), (255, 0, 0), 2)

        for i, (x, y) in enumerate(person):
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)  # Yellow keypoints

        for (p1, p2) in SKELETON_CONNECTIONS:
            x1, y1 = map(int, person[p1])
            x2, y2 = map(int, person[p2])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        right_elbow_wrist = calculate_angle(person[5], person[7], person[9])
        right_shoulder_elbow = calculate_angle(person[6], person[5], person[7])
        left_elbow_wrist = calculate_angle(person[6], person[8], person[10])
        left_shoulder_elbow = calculate_angle(person[5], person[6], person[8])

        right_posture = classify_posture(right_shoulder_elbow, right_elbow_wrist)
        left_posture = classify_posture(left_shoulder_elbow, left_elbow_wrist)

        label = f"Left: {left_posture}, Right: {right_posture}"
        cv2.putText(frame, label, (int(bbox_x_min), int(bbox_y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        angles = {
            "Right Elbow-Wrist": right_elbow_wrist,
            "Right Shoulder-Elbow": right_shoulder_elbow,
            "Left Elbow-Wrist": left_elbow_wrist,
            "Left Shoulder-Elbow": left_shoulder_elbow,
        }
        angle_data.append((person_id, angles))

def process_video(input_video, output_video, pdf_filename):
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    angle_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []

        draw_skeleton_and_angles(frame, keypoints, angle_data)
        out.write(frame)

    cap.release()
    out.release()
    export_angles_to_pdf(angle_data, pdf_filename)

def export_angles_to_pdf(angle_data, filename):
    """Export time-based joint angle graphs for multiple persons stacked vertically."""
    with PdfPages(filename) as pdf:
        times = list(range(len(angle_data)))
        people_ids = sorted(set(person_id for person_id, _ in angle_data))
        num_persons = len(people_ids)
        num_stacks = (num_persons + 4) // 5  # Stack 5 persons per page

        for stack_idx in range(num_stacks):
            fig, axes = plt.subplots(nrows=min(5, num_persons - stack_idx * 5) * 2, ncols=1, figsize=(10, 4 * min(5, num_persons - stack_idx * 5)), sharex=True)
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            for i, person_id in enumerate(people_ids[stack_idx * 5:(stack_idx + 1) * 5]):
                right_shoulder_angles = [data[1].get("Right Shoulder-Elbow", 0) for data in angle_data if data[0] == person_id]
                left_shoulder_angles = [data[1].get("Left Shoulder-Elbow", 0) for data in angle_data if data[0] == person_id]
                right_elbow_angles = [data[1].get("Right Elbow-Wrist", 0) for data in angle_data if data[0] == person_id]
                left_elbow_angles = [data[1].get("Left Elbow-Wrist", 0) for data in angle_data if data[0] == person_id]

                axes[i * 2].plot(times[:len(right_shoulder_angles)], right_shoulder_angles, label=f"Person {person_id} Right Shoulder", color='blue')
                axes[i * 2].plot(times[:len(left_shoulder_angles)], left_shoulder_angles, label=f"Person {person_id} Left Shoulder", linestyle='dashed', color='red')
                axes[i * 2].set_ylabel("Shoulder Angle (degrees)")
                axes[i * 2].set_ylim(0, 360)
                axes[i * 2].legend()
                axes[i * 2].grid()

                axes[i * 2 + 1].plot(times[:len(right_elbow_angles)], right_elbow_angles, label=f"Person {person_id} Right Elbow", color='blue')
                axes[i * 2 + 1].plot(times[:len(left_elbow_angles)], left_elbow_angles, label=f"Person {person_id} Left Elbow", linestyle='dashed', color='red')
                axes[i * 2 + 1].set_ylabel("Elbow Angle (degrees)")
                axes[i * 2 + 1].set_ylim(0, 360)
                axes[i * 2 + 1].legend()
                axes[i * 2 + 1].grid()

            axes[-1].set_xlabel("Time (frames)")
            fig.suptitle("Joint Angles Over Time (Stacked View)")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    print(f"Angles exported to {filename}")



process_video(r"C:\Users\peter\Desktop\WORKING THESIS FILES\RESOURCES\ACTUAL SURVEY\MARCH 1\TestIndiv2.mp4",
              r"C:\Users\peter\Desktop\WORKING THESIS FILES\MODULAR_TESTING\Tests\Thresholding 2\outputs\nicesana.mp4",
              r"C:\Users\peter\Desktop\WORKING THESIS FILES\MODULAR_TESTING\Tests\Thresholding 2\outputs\angles_report_2_nice.pdf")




import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ultralytics import YOLO

# Load YOLOv8 Pose Model
model = YOLO('yolov8n-pose.pt')

# Define keypoint indices (COCO format)
EAR_R_IDX, EAR_L_IDX = 4, 3
SHOULDER_R_IDX, SHOULDER_L_IDX = 6, 5
WRIST_R_IDX, WRIST_L_IDX = 10, 9
HIP_R_IDX, HIP_L_IDX = 12, 11

# Thresholds (Adjusted based on testing)
RAISE_Y_THRESHOLD = 0.3  # Wrist significantly above shoulder
FORWARD_Y_THRESHOLD = 0.4  # Wrist significantly ahead of shoulder-hip axis
BACKWARD_Y_THRESHOLD = -0.4  # Wrist significantly behind shoulder-hip axis
SIDE_X_THRESHOLD = 0.4  # Wrist significantly extended sideways
REST_Y_THRESHOLD = 0.15  # Wrist close to the waist
WAIST_Y_THRESHOLD = 0.2  # Wrist positioned near the hip

# Function to process video and analyze movements
def process_video(video_path, output_video_path, output_pdf_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    movement_data = []
    frame_index = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy()
            if keypoints.shape[0] == 0:
                continue
            
            right_shoulder, right_wrist, right_hip = keypoints[0, SHOULDER_R_IDX], keypoints[0, WRIST_R_IDX], keypoints[0, HIP_R_IDX]
            left_shoulder, left_wrist, left_hip = keypoints[0, SHOULDER_L_IDX], keypoints[0, WRIST_L_IDX], keypoints[0, HIP_L_IDX]
            
            # Compute torso center (midpoint between shoulders)
            torso_center = (right_shoulder + left_shoulder) / 2
            
            # Normalize distances by shoulder width
            shoulder_width = np.linalg.norm(np.array(right_shoulder) - np.array(left_shoulder))
            if shoulder_width == 0:
                continue
            
            right_y_dist = (right_shoulder[1] - right_wrist[1]) / shoulder_width  # Using shoulder as reference
            right_x_dist = (right_wrist[0] - torso_center[0]) / shoulder_width  # Using torso center for sideward movement
            right_wrist_to_hip = abs(right_wrist[1] - right_hip[1]) / shoulder_width
            
            left_y_dist = (left_shoulder[1] - left_wrist[1]) / shoulder_width
            left_x_dist = (left_wrist[0] - torso_center[0]) / shoulder_width
            left_wrist_to_hip = abs(left_wrist[1] - left_hip[1]) / shoulder_width
            
            # Determine arm posture (Right Arm)
            if right_y_dist > RAISE_Y_THRESHOLD:
                right_posture = "Raising Arm"
            elif right_x_dist > SIDE_X_THRESHOLD:
                right_posture = "Extending Right"
            elif right_x_dist < -SIDE_X_THRESHOLD:
                right_posture = "Extending Left"
            elif right_y_dist > FORWARD_Y_THRESHOLD:
                right_posture = "Extending Forward"
            elif right_y_dist < BACKWARD_Y_THRESHOLD:
                right_posture = "Extending Backward"
            elif right_wrist_to_hip < WAIST_Y_THRESHOLD:
                right_posture = "Resting"
            else:
                right_posture = "Unknown"
            
            # Determine arm posture (Left Arm)
            if left_y_dist > RAISE_Y_THRESHOLD:
                left_posture = "Raising Arm"
            elif left_x_dist > SIDE_X_THRESHOLD:
                left_posture = "Extending Right"
            elif left_x_dist < -SIDE_X_THRESHOLD:
                left_posture = "Extending Left"
            elif left_y_dist > FORWARD_Y_THRESHOLD:
                left_posture = "Extending Forward"
            elif left_y_dist < BACKWARD_Y_THRESHOLD:
                left_posture = "Extending Backward"
            elif left_wrist_to_hip < WAIST_Y_THRESHOLD:
                left_posture = "Resting"
            else:
                left_posture = "Unknown"
            
            # Store movement data
            time_sec = frame_index / fps
            movement_data.append((time_sec, right_y_dist, left_y_dist, right_posture, left_posture))
            
            # Draw keypoints and annotate
            for keypoint in keypoints[0]:
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            cv2.putText(frame, f"Right Arm: {right_posture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Left Arm: {left_posture}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        out.write(frame)
        frame_index += 1
    
    cap.release()
    out.release()
    
    if not movement_data:
        print("No movement data collected. Check if the model detects keypoints correctly.")
        return
    
    print("Processing complete. Movements detected and labeled.")

# Run the processing
process_video(r"C:\Users\peter\Desktop\WORKING THESIS FILES\RESOURCES\ACTUAL SURVEY\MARCH 1\TestIndiv2.mp4",
              r"C:\Users\peter\Desktop\WORKING THESIS FILES\MODULAR_TESTING\Tests\Arm Movement Visualization\outputs\processed_video.mp4",
              r"C:\Users\peter\Desktop\WORKING THESIS FILES\MODULAR_TESTING\Tests\Arm Movement Visualization\outputs\movement_graph.pdf")

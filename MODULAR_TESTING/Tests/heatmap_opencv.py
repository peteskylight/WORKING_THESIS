import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO('yolov8n.pt')

# Ensure the model is using the CPU
# device = 'cpu'
# model.to(device)

# Initialize video capture
video_path = r"C:\Users\THESIS_WORKSPACE\Desktop\WORKING_THESIS\RESOURCES\Sample Vids\Shorter.mp4"
cap = cv2.VideoCapture(video_path)

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize dictionary to store tracked humans and their positions
tracked_humans = {}

# Function to update the dictionary with the current frame
def update_tracked_humans(tracked_humans, current_frame, detected_humans):
    for human_id, bbox in detected_humans.items():
        if human_id not in tracked_humans:
            tracked_humans[human_id] = {'positions': [], 'frames': []}
        tracked_humans[human_id]['positions'].append(bbox)
        tracked_humans[human_id]['frames'].append(current_frame)

# Function to draw gradient circles using Gaussian blur
def draw_solid_circle(frame, center, radius, intensity):
    color = (0, 0, 255)  # Red color
    cv2.circle(frame, center, radius, color, -1)

# Process video frames
current_frame = 0
ret, frame = cap.read()
if not ret:
    print("Error: Unable to read video")
    cap.release()
    exit()

# Initialize heatmap
heatmap = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

while cap.isOpened():
    results = model.track(frame, conf=0.5, persist=True, classes=0, iou=0.3, agnostic_nms=True, imgsz=(1088, 608))
    id_name_dict = results[0].names
    detected_humans = {}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.id is not None and box.xyxy is not None and box.cls is not None:
                track_id = int(box.id.tolist()[0])
                bbox = box.xyxy.tolist()[0]
                object_cls_id = box.cls.tolist()[0]
                object_cls_name = id_name_dict.get(object_cls_id, "unknown")
                if object_cls_name == "person":
                    detected_humans[track_id] = bbox
    
    update_tracked_humans(tracked_humans, current_frame, detected_humans)
    current_frame += 1
    ret, frame = cap.read()
    if not ret:
        break

# Release video capture
cap.release()

# Create heatmap visualization
for human_id, data in tracked_humans.items():
    for bbox in data['positions']:
        center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        max_radius = 20
        intensity = len(data['frames']) * 5  # Adjust intensity based on the number of frames
        draw_solid_circle(heatmap, center, max_radius, intensity)

# Display the heatmap
heatmap = cv2.resize(heatmap,  (720,480))
cv2.imshow("Heatmap", heatmap)
cv2.waitKey(0)
cv2.destroyAllWindows()

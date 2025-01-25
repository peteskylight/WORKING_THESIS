
import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO('yolov8n.pt')

# Initialize video capture
video_path = r"C:\Users\USER\Desktop\WORKING_THESIS\RESOURCES\Sample Vids\Shorter.mp4"
cap = cv2.VideoCapture(video_path)

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize dictionary to store tracked humans
tracked_humans = {}

# Function to update the dictionary with the current frame
def update_tracked_humans(tracked_humans, current_frame, detected_humans):
    for human_id in detected_humans:
        if human_id not in tracked_humans:
            tracked_humans[human_id] = {'start_frame': current_frame, 'end_frame': current_frame}
        else:
            tracked_humans[human_id]['end_frame'] = current_frame

# Process video frames
current_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(frame, conf=0.5, persist=True, classes=0, iou=0.3, agnostic_nms=True, imgsz=(1088, 608))[0]
    id_name_dict = results.names
    student_dict = {}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.id is not None and box.xyxy is not None and box.cls is not None:
                track_id = int(box.id.tolist()[0])
                track_result = box.xyxy.tolist()[0]
                object_cls_id = box.cls.tolist()[0]
                object_cls_name = id_name_dict.get(object_cls_id, "unknown")
                if object_cls_name == "person":
                    student_dict[track_id] = track_result
                    if track_id not in tracked_humans:
                        tracked_humans[track_id] = {'start_frame': current_frame, 'end_frame': current_frame}
                    else:
                        tracked_humans[track_id]['end_frame'] = current_frame
    
    current_frame += 1

# Release video capture
cap.release()

# Calculate the total duration for each tracked human in seconds
total_durations = {}
for human_id, frames in tracked_humans.items():
    if frames['start_frame'] is not None and frames['end_frame'] is not None:
        duration_frames = frames['end_frame'] - frames['start_frame'] + 1
        duration_seconds = duration_frames / fps
        if human_id in total_durations:
            total_durations[human_id] += duration_seconds
        else:
            total_durations[human_id] = duration_seconds

# Print the total duration for each tracked human
for human_id, duration_seconds in total_durations.items():
    print(f"Human {human_id} existed for {duration_seconds:.2f} seconds throughout the video.")

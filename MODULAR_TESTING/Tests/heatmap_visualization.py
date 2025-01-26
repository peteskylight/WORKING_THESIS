import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO('yolov8n.pt')

# Ensure the model is using the CPU
device = 'cpu'
model.to(device)

# Initialize video capture
video_path = r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\Examination Sample Videos\Shorter.mp4"
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

# Function to create a gradient circle
def create_gradient_circle(radius, color):
    gradient_circle = np.zeros((radius*2, radius*2, 4), dtype=np.uint8)

    # Create gradient from center (fully opaque) to edges (fully transparent)
    for y in range(radius*2):
        for x in range(radius*2):
            distance = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)
            if distance < radius:
                alpha = 255 - int(255 * (distance / radius))
                gradient_circle[y, x] = [color[0], color[1], color[2], alpha]

    return gradient_circle

# Function to overlay image with alpha
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if no overlap
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, 3] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay_crop[:, :, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

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
        radius = 20
        intensity = len(data['frames']) * 5  # Adjust intensity based on the number of frames
        gradient_circle = create_gradient_circle(radius, (0, 0, 255))  # Red color
        overlay_image_alpha(heatmap, gradient_circle, (center[0] - radius, center[1] - radius), gradient_circle)

# Display the heatmap
cv2.imshow("Heatmap", heatmap)
cv2.waitKey(0)
cv2.destroyAllWindows()

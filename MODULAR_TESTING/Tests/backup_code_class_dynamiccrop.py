
import cv2
import numpy as np
from ultralytics import YOLO

class VideoProcessor:
    def __init__(self, video_frames, crop_height, crop_start_y, model_path, confidence_threshold):
        self.video_frames = video_frames
        self.crop_height = crop_height
        self.crop_start_y = crop_start_y
        self.human_detection_model = YOLO(model_path)
        self.human_detection_confidence = confidence_threshold
        self.student_detections_dicts = []
        self.human_detection_progress_update = lambda x: print(f"Progress: {x}%")
        self.human_track_results = lambda x: print("Tracking complete.")

    def create_roi_mask(self, frame):
        height, width, _ = frame.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        # Define ROI: First 3 columns from bottom to middle
        col_width = width // 7  # Assume 7 columns
        row_height = height // 6  # Assume 6 rows
        roi = (0, row_height * 3, col_width * 3, height)

        # Create a filled rectangle for ROI
        cv2.rectangle(mask, (roi[0], roi[1]), (roi[2], roi[3]), 255, -1)
        return mask

    def apply_mask(self, frame, mask):
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return masked_frame

    def run(self):
        total_frames = len(self.video_frames)
        current_frame = 0

        for frame in self.video_frames:
            # Step 1: Create and apply ROI mask
            roi_mask = self.create_roi_mask(frame)
            masked_frame = self.apply_mask(frame, roi_mask)

            # Step 2: Crop the frame to the required height
            crop_height = self.crop_height
            start_y = self.crop_start_y
            cropped_frame = masked_frame[start_y:start_y + crop_height, :]

            # Step 3: Use the YOLO model with tracking
            results = self.human_detection_model.track(
                cropped_frame,
                conf=self.human_detection_confidence,
                persist=True,
                classes=0,
                iou=0.3,
                agnostic_nms=True
            )[0]

            id_name_dict = results.names
            student_dict = {}

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box.id is not None and box.xyxy is not None and box.cls is not None:
                        track_id = int(box.id.tolist()[0])
                        track_result = box.xyxy.tolist()[0]
                        # Adjust bounding box coordinates relative to the original frame
                        track_result[1] += start_y
                        track_result[3] += start_y
                        object_cls_id = box.cls.tolist()[0]
                        object_cls_name = id_name_dict.get(object_cls_id, "unknown")
                        if object_cls_name == "person":
                            student_dict[track_id] = track_result
                    else:
                        print("One of the attributes is None:", box.id, box.xyxy, box.cls)

            self.student_detections_dicts.append(student_dict)

            # Update progress
            current_frame += 1
            progress = int((current_frame / total_frames) * 100)
            self.human_detection_progress_update(progress)

        # Emit the final tracking results
        self.human_track_results(self.student_detections_dicts)

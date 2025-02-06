import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator

import random
from PySide6.QtCore import QThread, Signal
import numpy as np
from ultralytics.utils.plotting import Annotator

class DrawingUtils:
    def __init__(self) -> None:
        pass

    def draw_bounding_box(self, frame, box):
        # Draw bounding box
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, "Tester", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def draw_bounding_box_import(self, frame, bbox, track_id):
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        cv2.putText(frame, f"Student ID: {track_id}", (int(bbox[0]), int(bbox[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    def drawPoseLandmarks(self, frame, keypoints):
        for keypoint in keypoints:
            x = int(keypoint[0] * frame.shape[1])
            y = int(keypoint[1] * frame.shape[0])
            cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
        
        return frame
    
    
    def draw_keypoints_and_skeleton(self,frame, keypoints):
        skeleton_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), 
        (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 12), 
        (11, 13), (12, 14), (13, 15), (14, 16)
        ]
    
        # Draw skeleton
        for pair in skeleton_pairs:
            pt1 = keypoints[pair[0]]
            pt2 = keypoints[pair[1]]
            x1 = int(pt1[0] * frame.shape[1])
            y1 = int(pt1[1] * frame.shape[0])
            x2 = int(pt2[0] * frame.shape[1])
            y2 = int(pt2[1] * frame.shape[0])
            if 0.1 <= x1 < frame.shape[1] and 0.1 <= y1 < frame.shape[0] and 0.1 <= x2 < frame.shape[1] and 0.1 <= y2 < frame.shape[0]:
                cv2.line(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        
        return frame



class HeatmapWorker(QThread):
    heatmap_ready = Signal(np.ndarray)  # Signal to send processed heatmap

    def __init__(self, seat_plan_picture):
        super().__init__()
        self.frame = None
        self.results = None
        self.running = True
        self.seat_plan_picture = seat_plan_picture.copy()

    def run(self):
        while self.running:
            if self.frame is not None and self.results is not None:
                heatmap = self.drawing_classroom_heatmap(self.frame, self.results)
                self.heatmap_ready.emit(heatmap)  # Send result back to main thread
                self.frame = None  # Reset frame to avoid unnecessary recomputation

    def process_frame(self, frame, results):
        """Receives a new frame and results, and processes it in the thread."""
        self.frame = frame.copy()
        self.results = results

    def stop(self):
        """Stops the worker thread."""
        self.running = False
        self.quit()
        self.wait()

    def drawing_classroom_heatmap(self, frame, results):
        heatmap_image = self.seat_plan_picture.copy()

        radius = 150
        gradient_circle = self.create_gradient_circle(radius, (255, 0, 0), 16)  # Precompute once

        for _, bbox in results.items():
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            self.overlay_image_alpha(heatmap_image, gradient_circle, (center[0] - radius, center[1] - radius))

        return heatmap_image

    def create_gradient_circle(self, radius, color, max_alpha):
        Y, X = np.ogrid[:2*radius, :2*radius]
        center = radius
        dist_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)

        alpha = np.clip(max_alpha - (max_alpha * (dist_from_center / radius)), 0, max_alpha).astype(np.uint8)

        gradient_circle = np.zeros((2*radius, 2*radius, 4), dtype=np.uint8)
        gradient_circle[..., :3] = color
        gradient_circle[..., 3] = alpha

        return gradient_circle

    def overlay_image_alpha(self, img, img_overlay, pos):
        x, y = pos
        h, w = img_overlay.shape[:2]

        y1, y2 = max(0, y), min(img.shape[0], y + h)
        x1, x2 = max(0, x), min(img.shape[1], x + w)

        y1o, y2o = max(0, -y), min(h, img.shape[0] - y)
        x1o, x2o = max(0, -x), min(w, img.shape[1] - x)

        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        alpha = img_overlay_crop[..., 3:4] / 255.0
        img[y1:y2, x1:x2, :3] = (alpha * img_overlay_crop[..., :3] +
                                  (1 - alpha) * img[y1:y2, x1:x2, :3]).astype(np.uint8)

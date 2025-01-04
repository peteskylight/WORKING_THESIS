import cv2
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

    def drawPoseLandmarks(self, frame, keypoints):
        for keypoint in keypoints:
            x = int(keypoint[0] * frame.shape[1])
            y = int(keypoint[1] * frame.shape[0])
            cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
        
        return frame
    
    
    def draw_keypoints_and_skeleton(self,frame, keypoints):
        skeleton_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6), 
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
    
    



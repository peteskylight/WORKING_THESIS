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

    def draw_bounding_box_import(self, frame, box, color):
        # Draw bounding box
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, "Tester", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color , 2)
    
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
    






class DrawingBoundingBoxesThread(QThread):
    frame_drawn_list = Signal(object)
    progress_updated = Signal(int)

    def __init__(self, results, white_frames):
        super().__init__()
        
        self.drawing_utils = DrawingUtils()
        self.results = results
        self.white_frames_list = white_frames
        self._running = True

    def run(self):
        total_frames = len(self.results)
        current_frame = 0
        frames = []
        for result, white_frame in zip(self.results, self.white_frames_list):
            if not self._running:
                break

            annotator = Annotator(white_frame)
            for boxes in result:
                # Generate a random color for each box
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                #Draw in White Frames
                self.drawing_utils.draw_bounding_box_import(frame=white_frame, box=boxes, color=color)
            
            frames.append(white_frame)
            
            current_frame += 1
            progress = int((current_frame / total_frames) * 100)
            self.progress_updated.emit(progress)
            
        self.frame_drawn_list.emit(frames)
        
    def stop(self):
        self._running = False
        self.wait()
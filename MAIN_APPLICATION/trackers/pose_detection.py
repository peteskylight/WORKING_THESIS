from ultralytics import YOLO
import cv2
import numpy as np
import sys
import os

from PySide6.QtCore import QThread, Signal
sys.path.append('../')

from ultralytics.utils.plotting import Annotator
from utils import DrawingUtils

class PoseDetection:
    def __init__(self, humanDetectionModel, humanDetectConf, humanPoseModel, humanPoseConf):
        self.human_detection_model = YOLO(humanDetectionModel)
        self.human_detection_conf = humanDetectConf
        self.human_pose_model = YOLO(humanPoseModel)
        self.human_pose_conf = humanPoseConf
    
        self.drawing_utils = DrawingUtils()

    def getResults(self, frame, model, confidenceRate):
        # Recolor Feed from RGB to BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Perform inference using the YOLO model
        results = model(frame, conf = confidenceRate, classes=0, iou=0.5, agnostic_nms=True)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return results
    
    def flatten_keypoints(self, human_pose_results):
        normalized_keypoints =  np.array(human_pose_results[0].keypoints.xyn.cpu().numpy()[0])
        flattenedKeypoints = normalized_keypoints.flatten()
        return flattenedKeypoints, normalized_keypoints
    
    def getHumanPoseKeypoints(self, frame):
        
        return_bbox = None
        
        human_pose_results = self.getModel(frame = frame,
                                            model = self.human_pose_model,
                                            confidenceRate = self.human_pose_conf)
        
        flatten_keypoints, normalized_keypoints = self.flatten_keypoints(human_pose_results=human_pose_results)

        for result in human_pose_results:
            bboxes = result.boxes.xyxy
            
            for bbox in bboxes:
                return_bbox = bbox
        
        return frame, normalized_keypoints, return_bbox



class PoseDetectionThread(QThread):
    human_detection_results = Signal(object)
    human_detection_progress_update = Signal(int)

    def __init__(self, video_frames,main_window, humanDetectionModel, humanDetectConf, humanPoseModel, humanPoseConf):
        super().__init__()
        self.main_window = main_window
        
        self.video_frames = video_frames
        self.human_detection_model = YOLO(humanDetectionModel)
        self.human_detection_confidence = humanDetectConf
        self.human_pose_model = YOLO(humanPoseModel)
        self.human_pose_confidence = humanPoseConf
        
        self.detection = PoseDetection(humanDetectionModel, humanDetectConf, humanPoseModel, humanPoseConf)
        self._running = True

    def run(self):
        total_frames = len(self.video_frames)
        current_frame = 0
        results_list = []
        
        
        for frame in self.video_frames:
            results = self.detection.getResults(frame=frame,
                                                model=self.human_detection_model,
                                                confidenceRate=self.human_detection_confidence)
            results_list.append(results)
            current_frame += 1
            progress = int((current_frame / total_frames) * 100)
            self.human_detection_progress_update.emit(progress)
            
            del results
            del progress
        
        self.human_detection_results.emit(results_list)
        
        del results_list

    def stop(self):
        self._running = False
        self.wait()

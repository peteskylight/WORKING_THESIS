from ultralytics import YOLO
import cv2
import numpy as np
import sys
import os
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

    
    def getModel(self, frame, model, confidenceRate):
        # Recolor Feed from RGB to BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Perform inference using the YOLO model
        results = model(frame, conf = confidenceRate)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return results
    
    def flatten_keypoints(self, human_pose_results):
        normalized_keypoints =  np.array(human_pose_results[0].keypoints.xyn.cpu().numpy()[0])
        flattenedKeypoints = normalized_keypoints.flatten()
        return flattenedKeypoints, normalized_keypoints
    
    def getHumanPoseKeypoints(self, frame):
        
        # Perform inference using the YOLO model
        human_results = self.getModel(frame=frame,
                                   model=self.human_detection_model,
                                   confidenceRate=self.human_detection_conf)
        
        student_dict = []
        
        for result in human_results:
            boxes = result.boxes
            for box in boxes:
                # Get the image per person
                b = box.xyxy[0]
                c = box.cls
                
                cropped_image = frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                human_pose_results = self.getModel(frame = cropped_image,
                                                   model = self.human_pose_model,
                                                   confidenceRate = self.human_pose_conf)
                
                flatten_keypoints, normalized_keypoints = self.flatten_keypoints(human_pose_results=human_pose_results)
                
                self.drawing_utils.drawPoseLandmarks(cropped_image=cropped_image,
                                                     normalized_keypoints = normalized_keypoints)
        
        return frame
                
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


    def getHumanTrackResults(self, frame, model, confidenceRate):
        # Recolor Feed from RGB to BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Perform inference using the YOLO model
        results = model.track(image,
                              conf = confidenceRate,
                              persist=True,
                              classes=0,
                              iou=0.5,
                              agnostic_nms=True)[0]
        id_name_dict = results.names

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return results, id_name_dict
    
    def flatten_keypoints(self, human_pose_results):
        normalized_keypoints =  np.array(human_pose_results[0].keypoints.xyn.cpu().numpy()[0])
        flattenedKeypoints = normalized_keypoints.flatten()
        return flattenedKeypoints, normalized_keypoints
    


class HumanDetectionThread(QThread):
    human_track_results = Signal(object)
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
        student_detections_dicts = []
        
        
        for frame in self.video_frames:
            
            
            results = self.human_detection_model.track(frame,
                                                  conf = self.human_detection_confidence,
                                                  persist=True,
                                                  classes=0,
                                                  iou=0.3,
                                                  agnostic_nms=True,
                                                  imgsz = (1088, 608))[0]
        
            id_name_dict = results.names
            
            student_dict = {}
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    #Get the image per person
                    b = box.xyxy[0]
                    c = box.cls
                    cropped_image = frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                    
                    if box.id is not None and box.xyxy is not None and box.cls is not None:
                        track_id = int(box.id.tolist()[0])
                        track_result = box.xyxy.tolist()[0]
                        object_cls_id = box.cls.tolist()[0]
                        object_cls_name = id_name_dict.get(object_cls_id, "unknown")
                        if object_cls_name == "person":
                            student_dict[track_id] = track_result
                    else:
                        print("One of the attributes is None:", box.id, box.xyxy, box.cls)
            
            student_detections_dicts.append(student_dict)
                
            current_frame += 1
            progress = int((current_frame / total_frames) * 100)
            self.human_detection_progress_update.emit(progress)
        
        self.human_track_results.emit(student_detections_dicts)
        

    def stop(self):
        self._running = False
        self.wait()


class PoseDetectionThread(QThread):
    pose_detection_results = Signal(object)
    pose_detection_progress_update = Signal(object)
    
    def __init__(self,original_frames, human_detect_results, humanDetectionModel, humanDetectConf, humanPoseModel, humanPoseConf):
        super().__init__()
        
        self.human_detection_model = YOLO(humanDetectionModel)
        self.human_detection_confidence = humanDetectConf
        self.human_pose_model = YOLO(humanPoseModel)
        self.human_pose_confidence = humanPoseConf
        
        self.original_frames = original_frames
        self.human_detect_results = human_detect_results
        self._running = True
    
    def run(self):
        total_frames = len(self.human_detect_results)
        current_frame = 0
        pose_results_list = []
        
        for frame, detection in zip(self.original_frames, self.human_detect_results):
            keypoints_dict = {}
            for track_id, bbox in detection.items():
                # Extract the bounding box coordinates
                x1, y1, x2, y2 = map(int, bbox)
                cropped_image = frame[y1:y2, x1:x2]
            try:
                print("OK")
                result = self.human_pose_model(cropped_image, self.human_pose_confidence)
                result_list = list(result)
                if result_list:
                    poseResults = result_list[0]
                    if poseResults.keypoints:
                        keypoints_normalized = np.array(poseResults[0].keypoints.xyn.cpu().numpy()[0])
                        keypoints_dict[track_id] = keypoints_normalized
                    else:
                        keypoints_dict[track_id] = np.zeros((17, 3))
                else:
                    print("Zero results")
                    keypoints_dict[track_id] = np.zeros((17, 3))
            except Exception as e:
                print("Zero results")
                keypoints_dict[track_id] = np.zeros((17, 3))
            
            pose_results_list.append(keypoints_dict)
            
            current_frame += 1
            progress = int((current_frame / total_frames) * 100)
            self.pose_detection_progress_update.emit(progress)
            
        
        self.pose_detection_results.emit(pose_results_list)
        #print(pose_results_list)
        
    def stop(self):
        self._running = False
        self.wait()

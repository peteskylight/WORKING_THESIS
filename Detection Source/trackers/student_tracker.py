
from ultralytics import YOLO
import cv2
import pickle
import numpy as np
import sys
import os
sys.path.append('../')

from ultralytics.utils.plotting import Annotator
from utils import DrawingUtils

class StudentTracker:
    def __init__(self, humanDetectionModel, humanDetectConf, humanPoseModel, humanPoseConf):
        self.humanDetectModel = YOLO(humanDetectionModel)
        self.humanPoseModel = YOLO(humanPoseModel)
        self.humanDetectConf = humanDetectConf
        self.humanPoseConf = humanPoseConf
    
        
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        student_detections = []
        pose_detections = []

        # Check if the directory exists, if not, create it
        if stub_path is not None:
            directory = os.path.dirname(stub_path)
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    print(f"Directory {directory} created.")
                except OSError as e:
                    print(f"Error creating directory {directory}: {e}")
                    return None
                

        if read_from_stub and stub_path is not None:
            try:
                with open(stub_path, 'rb') as f:
                    student_detections = pickle.load(f)
                print("Loaded data from stub file.")
                return student_detections
            except FileNotFoundError as e:
                print(f"Error loading from stub file: {e}")
                return None
            except Exception as e:
                print(f"An error occurred while loading from stub file: {e}")
                return None

        for frame in frames:
            student_dict= self.trackHuman(frame)
            student_detections.append(student_dict)

            # cv2.imshow("TEST", frame)
            # cv2.waitKey(10)
            
        if stub_path is not None:
            try:
                with open(stub_path, 'wb') as f:
                    pickle.dump((student_detections), f)
                print(f"Data saved to {stub_path}.")
            except OSError as e:
                print(f"Error saving to stub file: {e}")
                return None
            except Exception as e:
                print(f"An error occurred while saving to stub file: {e}")
                return None
        print("Student Detections Data Type: ", type(student_detections))
        return student_detections

    
    def detectHumanPose(self, frame, confidenceRate):

        # Perform inference using the YOLO model
        results = self.humanPoseModel(frame, conf = self.humanPoseConf)

        return frame, results

    def trackHuman(self, frame, confidenceRate=0.3):
        # Perform inference using the YOLO model
        results = self.humanDetectModel.track(frame, conf = self.humanDetectConf, persist=True, classes=0, iou=0.5, agnostic_nms=True)[0]
        
        id_name_dict = results.names
        
        student_dict = {}
        pose_results = []
        
        for result in results:
            boxes = result.boxes
            per_frame=[]
            for box in boxes:
                #Get the image per person
                b = box.xyxy[0]
                c = box.cls
                cropped_image = frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                # try:     
                #     _, poseResults = self.detectHumanPose(cropped_image, self.humanPoseConf)
                #     keypoints_normalized = np.array(poseResults[0].keypoints.xyn.cpu().numpy()[0])
                #     flattenedKeypoints = DrawingUtils.drawLandmarks(cropped_image, poseResults)
                #     #Track ID
                    
                # except Exception as e:
                #     print(e)
                    
                if box.id is not None and box.xyxy is not None and box.cls is not None:
                    track_id = int(box.id.tolist()[0])
                    track_result = box.xyxy.tolist()[0]
                    object_cls_id = box.cls.tolist()[0]
                    object_cls_name = id_name_dict.get(object_cls_id, "unknown")
                    if object_cls_name == "person":
                        student_dict[track_id] = track_result
                else:
                    print("One of the attributes is None:", box.id, box.xyxy, box.cls)
                    
            pose_results.append(per_frame)
            
        return student_dict #TAKE NOTE
    


    import numpy as np

    def detect_keypoints(self, frames, student_dicts):
        keypoints_dicts = []

        for frame, student_dict in zip(frames, student_dicts):
            keypoints_dict = {}
            for track_id, bbox in student_dict.items():
                # Extract the bounding box coordinates
                x1, y1, x2, y2 = map(int, bbox)
                cropped_image = frame[y1:y2, x1:x2]

                try:
                    # Perform pose detection on the cropped image
                    _, poseResults = self.detectHumanPose(cropped_image, self.humanPoseConf)
                    keypoints_normalized = np.array(poseResults[0].keypoints.xyn.cpu().numpy()[0])
                    
                    keypoints_dict[track_id] = keypoints_normalized
                except Exception as e:
                    print(f"Error detecting keypoints for track ID {track_id}: {e}")

            keypoints_dicts.append(keypoints_dict)

        return keypoints_dicts


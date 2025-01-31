import cv2
import argparse
import numpy as np
import os
import torch
from collections import defaultdict

from PySide6.QtCore import QThread, Signal
 
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
#from ultralytics.yolo.v8.detect.predict import Detections

from tensorflow.keras.models import load_model


import torch #========================================> GPU IMPORTANT <========


class ActionDetectionThread(QThread):
    processed_frames_list = Signal(object)
    processed_black_frames_list = Signal(object)
    detected_actions_list = Signal(object)
    progress_update = Signal(int)

    def __init__(self, video_keypoints, black_frames, video_frames, detections):
        super().__init__()

        self.video_keypoints = video_keypoints
        self.black_frames = black_frames
        self.video_frames = video_frames
        self.detections = detections

        self.action_recognition_model = load_model("RESOURCES/action_recognition_model.h5")
        self.temp_sequence = []
        self.buffer_size = 30
        self.actions_list = np.array(['Looking Down', 'Looking Forward', 'Looking Left', 'Looking Right', 'Looking Up']) 
        self.recent_action = None
        self.translate_action_results = None
        self.person_keypoints = defaultdict(lambda: [])
        self._running = True
    
    def preprocess_keypoints(self, keypoints_dict):
        """
        Convert keypoints dictionary into a properly shaped numpy array for inference.
        Missing keypoints (zeros) remain unchanged.
        """
        processed_data = {}
        for person_id, keypoints in keypoints_dict.items():
            keypoints = keypoints.flatten()  # Convert (17, 2) to (34,)
            processed_data[person_id] = keypoints
        return processed_data

    def draw_action_text(self, frame, black_frames, detections, actions):
        """
        Draw bounding boxes with action labels on each frame.
        """
        for person_id, bbox in detections.items():
            x, y, w, h = map(int, bbox)  # Ensure coordinates are integers
            action_text = f"Action: {actions.get(person_id, 'Unknown')}"
            
            # Ensure text position is within frame bounds
            text_x, text_y = x, max(y - 10, 10)  # Prevent negative y
            
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, action_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(black_frames, action_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, black_frames


    def predict_actions(self, frames_keypoints):
        """
        Predict actions based on sequential keypoints data.
        Returns a dictionary with person ID as keys and their predicted actions as values.
        """
        predictions = {}
        for person_id, keypoints_sequence in frames_keypoints.items():
            if len(keypoints_sequence) == self.buffer_size:
                input_data = np.array([keypoints_sequence])  # Shape (1, 30, 34)
                prediction = self.action_recognition_model.predict(input_data)[0]  # Assuming single output per person
                predicted_action = np.argmax(prediction)  # Get the action INDEX

                self.translate_action_results = self.actions_list[predicted_action] #GET THE ACTION NAME BASED ON THE INDEX!
                predictions[person_id] = self.translate_action_results
        return predictions
    
    def run(self):

        """
        Process video frames and return a list of frames with bounding boxes and action labels.
        """
        processed_frames = []
        processed_black_frames = []
        frame_count = len(self.video_keypoints) - 1

        for frame_idx, frame_keypoints in enumerate(self.video_keypoints):
            processed_keypoints = self.preprocess_keypoints(frame_keypoints)
            
            for person_id, keypoints in processed_keypoints.items():
                self.person_keypoints[person_id].append(keypoints)
                
                # Keep only the latest 30 frames
                if len(self.person_keypoints[person_id]) > self.buffer_size:
                    self.person_keypoints[person_id].pop(0)
            
            # Predict actions once we have enough frames
            action_predictions = self.predict_actions(self.person_keypoints)
            
            # Draw action labels on the frame
            frame = self.video_frames[frame_idx]
            black_frame = self.black_frames[frame_idx]
            frame, black_frame = self.draw_action_text(frame, black_frame, self.detections[frame_idx], action_predictions)
            
            progress = int((frame_idx / frame_count) * 100)
            self.progress_update.emit(progress)
            processed_frames.append(frame)
            processed_black_frames.append(black_frame)
        

        self.processed_frames_list.emit(processed_frames)
        self.processed_black_frames_list.emit(processed_black_frames)

    def stop(self):
        self._running = False
        self.wait()
import cv2
import argparse
import numpy as np
import os
import torch
 
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
#from ultralytics.yolo.v8.detect.predict import Detection

from tensorflow.keras.models import load_model


import torch #========================================> GPU IMPORTANT <========


class ActionDetection:
    def __init__(self):
        self.action_recognition_model = load_model("RESOURCES/action_recognition_model.h5")
        self.temp_sequence = []
        self.sequence_length = 30
        self.actions_list = np.array(['Looking Down', 'Looking Forward', 'Looking Left', 'Looking Right', 'Looking Up']) 
        self.recent_action = None
        self.translate_action_results = None


    

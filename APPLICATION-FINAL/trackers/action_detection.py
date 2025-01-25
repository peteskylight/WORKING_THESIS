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


class ActionDetectionThread:
    def __init__(self):
        pass

    

import cv2
from PySide2.QtCore import QTimer, Qt
from PySide2.QtGui import QImage, QPixmap

from trackers.pose_detection import PoseDetection
from utils.drawing_utils import DrawingUtils

class CameraFeed:
    def __init__(self, label):
        
        self.pose_detection = PoseDetection(humanDetectionModel='yolov8n.pt',
                                            humanDetectConf=0.4,
                                            humanPoseModel='yolov8n-pose.pt',
                                            humanPoseConf=0.4
                                            )
        self.drawing_utils = DrawingUtils()
        
        self.label = label
        self.cap = None
        self.timer = QTimer()

        self.timer.timeout.connect(self.update_frame)


    def start_camera(self, index):
        self.cap = cv2.VideoCapture(index)  
        self.timer.start(10) 

    def update_frame(self): #GET THE FRAME HERE
        ret, output_frame = self.cap.read()

        returned_frame = output_frame
        
        #==Conditional for
        returned_frame = self.pose_detection.getHumanPoseKeypoints(frame=output_frame)
        
        processed_frame = returned_frame
        
        if ret:
            # Convert the frame to QImage
            height, width, channel = processed_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(processed_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the QLabel
            self.label.setPixmap(QPixmap.fromImage(q_img))
        
        
        
    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()

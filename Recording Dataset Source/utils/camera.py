import cv2
import numpy as np
from PySide2.QtCore import QTimer, Qt
from PySide2.QtGui import QImage, QPixmap

from trackers.pose_detection import PoseDetection
from utils.drawing_utils import DrawingUtils

class CameraFeed:
    def __init__(self, label, white_frame_label, main_window):
        self.pose_detection = PoseDetection(humanDetectionModel='yolov8n.pt',
                                            humanDetectConf=0.4,
                                            humanPoseModel='yolov8n-pose.pt',
                                            humanPoseConf=0.4
                                            )
        self.drawing_utils = DrawingUtils()
        
        self.label = label
        self.white_frame_label = white_frame_label
        self.main_window = main_window
        self.cap = None
        self.timer = QTimer()

        self.timer.timeout.connect(self.update_frame)

    def start_camera(self, index):
        self.cap = cv2.VideoCapture(index)  
        self.timer.start(10) 

    def update_frame(self): #GET THE FRAME HERE
        ret, output_frame = self.cap.read()
        color = 255 #255 - WHITE, 0 - BLACK
        returned_frame = output_frame
        
        
        
        returned_frame, normalized_keypoints, bbox = self.pose_detection.getHumanPoseKeypoints(frame=output_frame)
        
        #==Conditional for
        if self.main_window.showCameraLandmarksChkBox.isChecked():
            self.drawing_utils.drawPoseLandmarks(frame = returned_frame,
                                                keypoints = normalized_keypoints)
        
        if self.main_window.showCameraBoundingBoxChkBox.isChecked():
            self.drawing_utils.draw_bounding_box(frame=returned_frame,
                                                 box=bbox)
        
        if self.main_window.show_skeleton_camera.isChecked():
            self.drawing_utils.draw_keypoints_and_skeleton(frame=returned_frame,
                                                           keypoints=normalized_keypoints)
        
        processed_frame = returned_frame
        
        if self.main_window.darkMode_whiteframe.isChecked():
            color = 0
        else:
            color = 255
        white_frame = color * np.ones_like(processed_frame)
        
        self.drawing_utils.drawPoseLandmarks(frame = white_frame,
                                                keypoints = normalized_keypoints)
        
        if self.main_window.show_whiteframe_boundingbox.isChecked():
            self.drawing_utils.draw_bounding_box(frame=white_frame,
                                                 box=bbox)
        
        if self.main_window.show_skeleton_white_frame.isChecked():
            self.drawing_utils.draw_keypoints_and_skeleton(frame=white_frame,
                                                 keypoints=normalized_keypoints)
            
        
        if ret:
            # Convert the frame to QImage
            height, width, channel = processed_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(processed_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the QLabel with aspect ratio maintained and white spaces
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)

            # Generate a white frame
            
            white_q_img = QImage(white_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            white_pixmap = QPixmap.fromImage(white_q_img)
            scaled_white_pixmap = white_pixmap.scaled(self.white_frame_label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.white_frame_label.setPixmap(scaled_white_pixmap)

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            
        # Clear the labels and set the text
        self.label.clear()
        self.label.setText("Camera stopped. No feed available.")
        self.label.setAlignment(Qt.AlignCenter)
        
        self.white_frame_label.clear()
        self.white_frame_label.setText("White frame stopped. No feed available.")
        self.white_frame_label.setAlignment(Qt.AlignCenter)

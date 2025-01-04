import cv2
import os
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
import numpy as np
from PySide2.QtCore import QTimer, Qt
from PySide2.QtGui import QImage, QPixmap

from trackers.pose_detection import PoseDetection
from utils.drawing_utils import DrawingUtils
from utils.tools import Tools

class CameraFeed:
    def __init__(self, label, white_frame_label, main_window):
        self.pose_detection = PoseDetection(humanDetectionModel='yolov8n.pt',
                                            humanDetectConf=0.4,
                                            humanPoseModel='yolov8n-pose.pt',
                                            humanPoseConf=0.4
                                            )
        self.drawing_utils = DrawingUtils()
        self.tools_utils = Tools()
        
        self.isRecording = False
        self.folder_count = 0
        self.frame_count = 0
        self.alreadyChecked = False
    
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
        
        #==Conditionals for VISUALIZATIONS
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
            
        #===FOR LOGGING
        
        if self.main_window.recording_button.text() == "STOP\nRECORDING":
            chosen_directory = self.main_window.directoryLineEdit.text()
            chosen_action = self.main_window.action_comboBox.currentText()
            destination_directory = os.path.join(chosen_directory, chosen_action)
            
            self.folder_count = self.tools_utils.count_folders(directory=destination_directory)
            
            # Ensure the folder exists before recording keypoints
            if not os.path.isdir(os.path.join(destination_directory, str(self.folder_count))):
                os.mkdir(os.path.join(destination_directory, str(self.folder_count)))
            
            self.record_and_save_keypoints(normalized_keypoints=normalized_keypoints,
                                        frame_num=self.frame_count)
            
            self.frame_count += 1
            
            if self.frame_count % 30 == 0:
                self.frame_count = 0
                self.folder_count += 1
                os.mkdir(os.path.join(destination_directory, str(self.folder_count)))

        elif self.main_window.recording_button.text() == "START\nRECORDING":
           self.frame_count = 0
            
        #===FOR SHOWING IN LABELS
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

    
    def record_and_save_keypoints(self, normalized_keypoints, frame_num):
        flattenedList = normalized_keypoints.flatten()
        chosen_directory = self.main_window.directoryLineEdit.text()
        chosen_action = self.main_window.action_comboBox.currentText()
        destination_directory = os.path.join(chosen_directory, chosen_action)
        no_of_sequences = self.main_window.sequence_slider.value()
        
        if not os.path.isdir(destination_directory):
            QMessageBox.critical(self.main_window, "Error", "The specified directory does not exist. Check the chosen directory.")
            return
        
        final_destination_directory = os.path.join(destination_directory, str(self.folder_count))

        # NEW Export keypoints
        npy_path = os.path.join(final_destination_directory, str(frame_num))
        np.save(npy_path, flattenedList)
        
        






# import os
# import numpy as np
# from PySide2.QtWidgets import QMessageBox

# def check_and_prepare_folder(self):
#     chosen_directory = self.main_window.directoryLineEdit.text()
#     chosen_action = self.main_window.action_comboBox.currentText()
#     destination_directory = os.path.join(chosen_directory, chosen_action)
    
#     if not self.alreadyChecked:
#         if len(npy_files) < 30:
#             # Delete all npy files in the folder
#             for f in npy_files:
#                 os.remove(os.path.join(folder_path, f))
#             print("Deleted all npy files in the folder and reset frame_num to 0.")
#             self.frame_count = 0
#         self.alreadyChecked = True
#         return folder_count, 0  # Reset frame_num to 0
    
#     return folder_count, None  # No need to reset frame_num

# def record_and_save_keypoints(self, normalized_keypoints, frame_num):
#     flattenedList = normalized_keypoints.flatten()
#     chosen_directory = self.main_window.directoryLineEdit.text()
#     chosen_action = self.main_window.action_comboBox.currentText()
#     destination_directory = os.path.join(chosen_directory, chosen_action)
#     print("HOORAY!!!")
#     no_of_sequences = self.main_window.sequence_slider.value()
    
    
#     # Check if the folder with the name same as the folder count has 30 npy files
#     folder_path = os.path.join(destination_directory, str(folder_count))
#     npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
            
            
            
#     if not os.path.isdir(destination_directory):
#         QMessageBox.critical(self.main_window, "Error", "The specified directory does not exist. Check the chosen directory.")
#         return
    
#     # Increment folder count every 30 frames
#     folder_count = self.tools_utils.count_folders(directory=destination_directory)
#     if frame_num % 30 == 0:
#         folder_count += 1
    
#     final_destination_directory = os.path.join(destination_directory, str(folder_count))

#     # NEW Export keypoints
#     npy_path = os.path.join(final_destination_directory, str(frame_num))
#     np.save(npy_path, flattenedList)
#     frame_num += 1

# # Example usage
# if self.main_window.recording_button.text() == "STOP\nRECORDING":
#     folder_count, reset_frame_num = self.check_and_prepare_folder()
#     if reset_frame_num is not None:
#         frame_num = reset_frame_num
#     else:
#         frame_num = 2  # Replace with actual frame number

#     self.record_and_save_keypoints(normalized_keypoints=normalized_keypoints, frame_num=frame_num)

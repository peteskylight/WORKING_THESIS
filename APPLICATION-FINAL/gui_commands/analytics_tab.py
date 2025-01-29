import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


class AnalyticsTab:
    def __init__(self, main_window):
        self.main_window = main_window
        pass
    

    def update_frame_for_center_video_label(self, frame):
        if frame is not None and self.main_window.is_center_video_playing:
            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize the frame proportionally to a width of 551 pixels
            new_width = 551
            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_height = int(new_width / aspect_ratio)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Crop the frame in the middle to a size of 551x191 pixels
            crop_height = 191
            start_y = (new_height - crop_height) // 2
            cropped_frame = resized_frame[start_y:start_y + crop_height, :]
            
            # Convert the cropped frame to QImage
            height, width, channel = cropped_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(cropped_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the QLabel with aspect ratio maintained and white spaces
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.main_window.center_video_preview_label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.main_window.center_video_preview_label.setPixmap(scaled_pixmap)

    def update_frame_for_front_video_label(self, frame, starting_y):

        if frame is not None:
            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame_height, frame_width, _ = frame.shape
            
            # Define the starting y coordinate and height for cropping
              # Height of the cropped region
            y = starting_y  # Starting y-coordinate for cropping
            h = 1080-y  # Height of the cropped region

            # Ensure the cropping region is within the frame's bounds
            if y + h > frame_height:
                y = max(0, frame_height - h)  # Adjust y to keep the cropping region within bounds
                h = min(h, frame_height)      # Adjust h if necessary
            
            # Crop the frame
            cropped_frame = frame[y:y+h, 0:frame_width]
            
            # Convert the cropped frame to QImage
            height, width, channel = cropped_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(cropped_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the QLabel with aspect ratio maintained and white spaces
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.main_window.front_video_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.main_window.front_video_preview_label.setPixmap(scaled_pixmap)

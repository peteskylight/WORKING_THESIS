import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


class AnalyticsTab:
    def __init__(self, main_window):
        self.main_window = main_window
        pass
    

    def update_frame_for_center_video_label(self, frame, center_starting_y, front_starting_y):
        if frame is not None:
            # Get the dimensions of the frame
            frame_height, frame_width, _ = frame.shape

            # Debugging: Print frame dimensions and input parameters
            #print(f"Frame dimensions: {frame.shape}")
            #print(f"Input parameters - center_starting_y: {center_starting_y}, front_starting_y: {front_starting_y}")

            # Calculate the height of the cropped region
            h = front_starting_y - center_starting_y

            # Ensure the cropping region is within the frame's bounds
            if center_starting_y < 0:
                center_starting_y = 0  # Ensure y is not negative
            if front_starting_y > frame_height:
                front_starting_y = frame_height  # Ensure front_starting_y does not exceed frame height
            if h <= 0:
                raise ValueError("Invalid cropping parameters: front_starting_y must be greater than center_starting_y.")

            # Crop the frame
            cropped_frame = frame[center_starting_y:front_starting_y, 0:frame_width]

            # Debugging: Print the dimensions of the cropped frame
            #print(f"Cropped frame dimensions: {cropped_frame.shape}")

            # Convert the cropped frame to QImage
            height, width, channel = cropped_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(cropped_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the QLabel with aspect ratio maintained
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.main_window.center_video_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.main_window.center_video_preview_label.setPixmap(scaled_pixmap)

    def update_frame_for_front_video_label(self, frame, starting_y, whole_classroom_height):

        if frame is not None:
            frame_height, frame_width, _ = frame.shape

            # Define the starting y coordinate and height for cropping

            y = starting_y  # Starting y-coordinate for cropping
            h = whole_classroom_height-y  # Just base the height on the starting y-coordinate difference para hindi nakakatamad magsubtract lagi haha. Make it a parameter too.

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

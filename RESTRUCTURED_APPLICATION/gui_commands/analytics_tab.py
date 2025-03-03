import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox

from pathlib import Path


class AnalyticsTab:
    def __init__(self, main_window):
        self.main_window = main_window
        script_dir = Path(__file__).parent  # Get script's folder
        image_path = script_dir.parent / "assets" / "SEAT PLAN.png"
        self.seat_plan_picture = cv2.imread(str(image_path))
        
        self.action_labels = ['All Actions', 'Extending Right Arm', 'Standing', 'Sitting']
        
        # Create combo box for selecting actions
        self.action_selector = QComboBox()
        self.action_selector.addItems(self.action_labels)
        self.action_selector.currentIndexChanged.connect(self.update_selected_action)
        self.action_selector.currentIndexChanged.connect(self.update_selected_action)
        self.action_selector.setFixedSize(120, 30) 
        
        # Add combo box to the main window
        self.main_window.layout().addWidget(self.action_selector)
        
        self.selected_action = 'All Actions'  # Default selection

    def update_selected_action(self):
        """Updates the selected action from the combo box."""
        self.selected_action = self.action_selector.currentText()
        self.update_heatmap()

    def update_heatmap(self):
        """Updates the heatmap based on the selected action."""
        frame = self.generate_heatmap_based_on_action()
        
        if frame is None:
            return
        
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Set the QImage to the QLabel with aspect ratio maintained
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.main_window.heatmap_present_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.main_window.heatmap_present_label.setPixmap(scaled_pixmap)

    def generate_heatmap_based_on_action(self):
        """Generates the heatmap image based on the selected action."""
        # This function should process data and return the heatmap image
        # Modify this logic to filter the heatmap based on the selected action
        if self.selected_action == 'All Actions':
            return self.generate_full_heatmap()
        else:
            return self.generate_filtered_heatmap(self.selected_action)

    def generate_full_heatmap(self):
        """Returns the heatmap image with all actions visualized."""
        # Placeholder for heatmap generation logic
        return np.zeros((480, 640, 3), dtype=np.uint8)  # Example black image

    def generate_filtered_heatmap(self, action):
        """Returns the heatmap image filtered for the specific action."""
        # Placeholder for heatmap filtering logic based on action
        return np.zeros((480, 640, 3), dtype=np.uint8)  # Example black image
    
    def remove_extended_width(self, frame: np.ndarray, extension: int = 500) -> np.ndarray:
        """
        Removes the extended width from a frame by cropping the black padding.

        Parameters:
        frame (np.ndarray): The input frame with extended width.
        extension (int): The number of pixels to remove from each side. Default is 500.

        Returns:
        np.ndarray: The cropped frame without the extended width.
        """
        height, width, channels = frame.shape
        expected_width = 1920 + (2 * extension)
        
        # Ensure the input frame has the expected extended width
        if width != expected_width or height != 1080:
            raise ValueError("Expected a frame with extended width.")
        
        # Crop the frame to remove the black padding
        cropped_frame = frame[:, extension:extension + 1920]
        
        return cropped_frame

    def update_frame_for_center_video_label(self, frame, center_starting_y, front_starting_y):
        if frame is not None:
            # Remove extended width directly inside this function
            extension = 300
            if len(frame.shape) == 3:
                frame_height, frame_width, channels = frame.shape
            else:
                frame_height, frame_width = frame.shape

            if frame_width > extension * 2:
                frame = frame[:, extension:-extension]  # Remove extended width from both sides

            # Recalculate frame dimensions after cropping width
            if len(frame.shape) == 3:
                frame_height, frame_width, _ = frame.shape
            else:
                frame_height, frame_width = frame.shape

            h = front_starting_y - center_starting_y

            # Ensure the cropping region is within the frame's bounds
            center_starting_y = max(0, center_starting_y)  # Ensure y is not negative
            front_starting_y = min(frame_height, front_starting_y)  # Ensure front_starting_y does not exceed frame height

            if h <= 0:
                raise ValueError("Invalid cropping parameters: front_starting_y must be greater than center_starting_y.")


            # Convert the cropped frame to QImage
            # Crop the frame
            cropped_frame = np.ascontiguousarray(frame[center_starting_y:front_starting_y, 0:frame_width])

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
            # Remove extended width directly inside this function
            extension = 300
            if len(frame.shape) == 3:
                frame_height, frame_width, channels = frame.shape
            else:
                frame_height, frame_width = frame.shape

            if frame_width > extension * 2:
                frame = frame[:, extension:-extension]  # Remove extended width from both sides

            # Recalculate frame dimensions after cropping width
            if len(frame.shape) == 3:
                frame_height, frame_width, _ = frame.shape
            else:
                frame_height, frame_width = frame.shape

            # Define the starting y coordinate and height for cropping
            y = starting_y  # Starting y-coordinate for cropping
            h = whole_classroom_height - y  # Base height on the difference to avoid manual subtraction

            # Ensure the cropping region is within the frame's bounds
            if y + h > frame_height:
                y = max(0, frame_height - h)  # Adjust y to keep the cropping region within bounds
                h = min(h, frame_height)      # Adjust h if necessary

            # Crop the frame
            # Crop the frame
            cropped_frame = np.ascontiguousarray(frame[y:y + h, 0:frame_width])

            # Convert the cropped frame to QImage
            height, width, channel = cropped_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(cropped_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()


            # Set the QImage to the QLabel with aspect ratio maintained and white spaces
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.main_window.front_video_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.main_window.front_video_preview_label.setPixmap(scaled_pixmap)


    def update_heatmap(self, frame):
        height, width = frame.shape[:2]

        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Set the QImage to the QLabel with aspect ratio maintained and white spaces
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.main_window.heatmap_present_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.main_window.heatmap_present_label.setPixmap(scaled_pixmap)
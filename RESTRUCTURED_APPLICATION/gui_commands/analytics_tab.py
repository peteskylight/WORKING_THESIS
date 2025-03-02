import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget
from pathlib import Path


class AnalyticsTab:
    def __init__(self, main_window):
            self.main_window = main_window
            script_dir = Path(__file__).parent  # Get script's folder
            image_path = script_dir.parent / "assets" / "SEAT PLAN.png"
            self.seat_plan_picture = cv2.imread(str(image_path))
            
            self.selected_action = "Extending Right Arm"
            self.action_labels = ['Extending Right Arm', 'Standing', 'Sitting']
            
            # Create UI for action selection
            self.button_widget = QWidget()
            self.button_layout = QVBoxLayout()
            self.buttons = {}
            
            for label in self.action_labels:
                button = QPushButton(label)
                button.setCheckable(True)
                button.setChecked(label == self.selected_action)
                button.clicked.connect(self.create_button_callback(label))
                self.button_layout.addWidget(button)
                self.buttons[label] = button
            
            self.button_widget.setLayout(self.button_layout)
            self.main_window.heatmap_controls_layout.addWidget(self.button_widget)
    
    def create_button_callback(self, label):
        def callback():
            self.update_selected_action(label)
        return callback
    
    def update_selected_action(self, action):
        self.selected_action = action
        for label, button in self.buttons.items():
            button.setChecked(label == action)
        self.update_heatmap_display()
    
    def update_heatmap_display(self):
        print(f"Updating heatmap for action: {self.selected_action}")
        heatmap = self.seat_plan_picture.copy()
        
        grid_counts = {}
        
        for frame in self.main_window.human_detect_results_front:
            for person_id, action in frame.items():
                if action == self.selected_action:
                    if person_id not in grid_counts:
                        grid_counts[person_id] = 0
                    grid_counts[person_id] += 1
        
        max_count = max(grid_counts.values()) if grid_counts else 1
        
        for person_id, count in grid_counts.items():
            intensity = int((count / max_count) * 255)
            color = (0, 0, intensity)
            x, y, w, h = self.get_person_coordinates(person_id)
            cv2.rectangle(heatmap, (x, y), (x + w, y + h), color, -1)
        
        self.update_heatmap(heatmap)
    
    def get_person_coordinates(self, person_id):
        # Placeholder function for getting coordinates based on person_id
        # Replace this with the actual mapping logic
        x = (person_id * 20) % self.seat_plan_picture.shape[1]
        y = (person_id * 15) % self.seat_plan_picture.shape[0]
        w, h = 50, 50  # Example bounding box size
        return x, y, w, h
    
    def update_heatmap(self, frame):
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Set the QImage to the QLabel with aspect ratio maintained
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.main_window.heatmap_present_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.main_window.heatmap_present_label.setPixmap(scaled_pixmap)

    
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
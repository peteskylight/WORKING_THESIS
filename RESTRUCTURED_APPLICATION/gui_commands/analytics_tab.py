import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QWidget, QVBoxLayout, QTableWidget
from pathlib import Path

class AnalyticsTab:
    def __init__(self, main_window):
        self.main_window = main_window
        script_dir = Path(__file__).parent  # Get script's folder
        image_path = script_dir.parent / "assets" / "SEAT PLAN.png"
        self.seat_plan_picture = cv2.imread(str(image_path))

        self.selected_actions = set()  # Store selected actions
        self.action_labels = ["Sitting", "Standing", "Extending Right Arm"]

        # Create a layout to contain the heatmap and checkboxes
        self.heatmap_layout = QVBoxLayout()
        self.heatmap_layout.addWidget(self.main_window.heatmap_present_label)  # QLabel supports setPixmap, not addWidget

        # Create UI for action selection
        self.button_widget = QWidget()
        self.button_layout = QVBoxLayout()
        self.checkboxes = {}

        for label in self.action_labels:
            checkbox = QCheckBox(label)
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(self.create_checkbox_callback(label))
            self.button_layout.addWidget(checkbox)
            self.checkboxes[label] = checkbox

        self.button_widget.setLayout(self.button_layout)
        self.heatmap_layout.addWidget(self.button_widget)
        self.main_window.heatmap_container.setLayout(self.heatmap_layout)  # Assuming you have a container widget

    def create_checkbox_callback(self, label):
        def callback(state):
            if state == Qt.Checked:
                self.selected_actions.add(label)
            else:
                self.selected_actions.discard(label)
            self.update_heatmap_display()
        return callback

    def update_heatmap_display(self):
        print(f"Updating heatmap for actions: {self.selected_actions}")
        heatmap = self.seat_plan_picture.copy()
        grid_counts = {}

        for frame in self.main_window.human_detect_results_front:
            for person_id, action in frame.items():
                if action in self.selected_actions:  # Check if action is selected
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

    def update_heatmap(self, frame):
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Set the QImage to the QLabel
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
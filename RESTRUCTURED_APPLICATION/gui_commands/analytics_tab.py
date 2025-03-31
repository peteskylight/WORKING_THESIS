import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox
from utils.video_utils import SeekingVideoPlayerThread
from pathlib import Path


class AnalyticsTab:
    def __init__(self, main_window, action_results_front, action_results_center, human_detect_results_front, human_detect_results_center, front_video_path, center_video_path):
        self.main_window = main_window
        self.Action = self.main_window.Action  # Get the ComboBox
        selected_action = None
        filtered_bboxes_front = {}
        filtered_bboxes_center = {}
        frame = {}
        self.video_utils = SeekingVideoPlayerThread(center_video_path, front_video_path, main_window, selected_action, filtered_bboxes_front, filtered_bboxes_center)
        self.merged_results = {}
        self.action_results_list_front = action_results_front
        self.action_results_list_center = action_results_center
       

        
        # Load Seat Plan Image (for resetting heatmap)
        script_dir = Path(__file__).parent
        image_path = script_dir.parent / "assets" / "SEAT PLAN.png"
        self.seat_plan_picture = cv2.imread(str(image_path))

        # Action filtering setup
        self.action_labels = ['All Actions', 'Extending Right Arm', 'Standing', 'Sitting']
        self.human_detect_results_front = human_detect_results_front
        self.human_detect_results_center = human_detect_results_center
        self.action_results_front = action_results_front
        self.action_results_center = action_results_center
        
        # Connect ComboBox to update function
        self.Action.currentIndexChanged.connect(self.update_selected_action)
        
        # Ensure initial heatmap frame exists
        if self.main_window.heatmap_frame is None:
            print("[ERROR] No initial heatmap frame found! Generating default heatmap...")
            self.main_window.heatmap_frame = self.generate_full_heatmap()

    def update_selected_action(self):
        """Handles action selection change and resets the heatmap instantly."""
        selected_action = self.Action.currentText()
        print(f"[DEBUG] Selected action changed to: {selected_action}. Resetting heatmap...")

        # Reset heatmap
        self.heatmap_image = self.seat_plan_picture.copy()
        self.main_window.heatmap_frame = self.heatmap_image.copy()
        self.main_window.heatmap_present_label.clear()

        # Ensure required data exists
        if not hasattr(self, 'action_results_list_front') or not hasattr(self, 'action_results_list_center'):
            print("[ERROR] Missing action detection results!")
            return

        # Extract filtered bounding boxes based on the selected action
        filtered_bboxes_front, filtered_bboxes_center = self.get_filtered_bboxes(
            selected_action=selected_action,
            action_results_list_front=self.action_results_list_front,
            action_results_list_center=self.action_results_list_center,
        )

        # Store filtered data in video_utils
        self.video_utils.filtered_bboxes_front = filtered_bboxes_front
        self.video_utils.filtered_bboxes_center = filtered_bboxes_center
        self.video_utils.selected_action = selected_action


        frame = self.heatmap_image.copy()
        self.video_utils.preload_frames()
        return frame

    def update_heatmap(self, frame): 
        """Updates the heatmap display without filtering."""

        if frame is None or frame.size == 0:
            print("[ERROR] Invalid frame received! Skipping heatmap update.")
            return

        self.main_window.heatmap_frame = frame.copy()
        print("[DEBUG] Heatmap frame updated successfully!")

        # **Step 3: Convert frame to QImage for QLabel display**
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # **Step 4: Display in QLabel**
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.main_window.heatmap_present_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.main_window.heatmap_present_label.setPixmap(scaled_pixmap)

    def get_filtered_bboxes(self, selected_action, action_results_list_front, action_results_list_center):
        """Filters bounding boxes based on the selected action indexes."""

        human_detect_results_front = self.human_detect_results_front 
        human_detect_results_center = self.human_detect_results_center

        filtered_bboxes_front = []
        filtered_bboxes_center = []

        # Normalize selected action (convert to lowercase for case-insensitive matching)
        selected_action = selected_action.strip().lower()

        # **Step 1: Find indexes where the selected action appears**
        indexes_front = [i for i, actions in enumerate(action_results_list_front) 
                        if selected_action in [a.lower() for a in actions.values()]]

        indexes_center = [i for i, actions in enumerate(action_results_list_center) 
                        if selected_action in [a.lower() for a in actions.values()]]

        # **Step 2: Extract bounding boxes using the found indexes**
        for i in indexes_front:
            if 0 <= i < len(human_detect_results_front):
                bbox_data = human_detect_results_front[i]  # Dictionary with numeric keys
                for key, bbox in bbox_data.items():  # Iterate over key-value pairs
                    if isinstance(bbox, list) and len(bbox) == 4:  # Check if bbox is valid
                        x1, y1, x2, y2 = bbox
                        filtered_bboxes_front.append((x1, y1, x2 - x1, y2 - y1))

        for i in indexes_center:
            if 0 <= i < len(human_detect_results_center):
                bbox_data = human_detect_results_center[i]
                for key, bbox in bbox_data.items():
                    if isinstance(bbox, list) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        filtered_bboxes_center.append((x1, y1, x2 - x1, y2 - y1))


        return filtered_bboxes_front, filtered_bboxes_center




    def generate_full_heatmap(self):
        """Generates a blank heatmap and stores it in heatmap_frame."""
        return self.seat_plan_picture.copy()  # Just return the seat plan image


    
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
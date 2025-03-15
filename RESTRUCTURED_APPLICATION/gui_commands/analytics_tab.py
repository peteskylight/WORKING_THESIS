import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox

from pathlib import Path


class AnalyticsTab:
    def __init__(self, main_window, action_results_front, action_results_center, human_detect_results_front, human_detect_results_center):
        self.main_window = main_window
        script_dir = Path(__file__).parent  # Get script's folder
        image_path = script_dir.parent / "assets" / "SEAT PLAN.png"
        self.seat_plan_picture = cv2.imread(str(image_path))
        self.action_selector = None
        
        self.action_labels = ['All Actions', 'Extending Right Arm', 'Standing', 'Sitting']
        self.human_detect_results_front = human_detect_results_front
        self.human_detect_results_center = human_detect_results_center
        self.action_results_front = action_results_front
        self.action_results_center = action_results_center
        self.action_results_list_front = []  # Initialize as empty list
        self.action_results_list_center = []  # Initialize as empty list

        


    def toggle_analytics_tab(self):
        # Toggle the visibility of the tab
        if self.analytics_tab_index == -1 or not self.MainTab.isTabVisible(self.analytics_tab_index):
            # Add the tab back
            self.MainTab.addTab(self.analytics_tab, self.analytics_tab_title)
            self.analytics_tab_index = self.MainTab.indexOf(self.analytics_tab)
            self.MainTab.setCurrentIndex(self.analytics_tab_index)

            self.action_selector = QComboBox()
            self.action_selector.addItems(self.action_labels)  # Now this works
            self.action_selector.currentIndexChanged.connect(self.update_selected_action)
            self.action_selector.setFixedSize(120, 30)
            
            
            self.selected_action = 'All Actions'  # Default selection
        else:
            # Remove the tab
            self.MainTab.removeTab(self.analytics_tab_index)
            self.analytics_tab_index = -1

    def update_selected_action(self):
        """Updates the selected action from the combo box and refreshes the heatmap."""
        if self.action_selector:
            self.selected_action = self.action_selector.currentText()
            print(f"Selected Action: {self.selected_action}")  # Debugging print

            # Get the latest heatmap frame
            heatmap_frame = self.get_latest_heatmap_frame()  

            if heatmap_frame is not None:
                print("[DEBUG] Heatmap frame received, updating...")
                
                self.update_heatmap(
                    heatmap_frame, 
                    self.selected_action, 
                    self.human_detect_results_front,  
                    self.human_detect_results_center,
                    self.action_results_list_front,  # Pass action results
                    self.action_results_list_center
                )
            else:
                print("[DEBUG] No heatmap frame available, generating a new one.")
                new_heatmap = self.generate_full_heatmap()
                self.update_heatmap(
                new_heatmap,
                self.selected_action,
                self.human_detect_results_front,
                self.human_detect_results_center,
                self.action_results_list_front,
                self.action_results_list_center
            )


    def update_heatmap(self, frame, selected_action, action_results_front, action_results_center, human_detect_results_front, human_detect_results_center):
        self.human_detect_results_center = human_detect_results_center
        self.human_detect_results_front = human_detect_results_front
       
        """Updates the heatmap based on the selected action and integrates action detection results."""
        if frame is None:
            print("[DEBUG] No frame provided for heatmap update.")
            return

        # Ensure action results are valid lists before proceeding
        if action_results_front is None:
            action_results_front = []

        if action_results_center is None:
            action_results_center = []

        # Filter data based on the selected action
        filtered_data = []

        if selected_action and selected_action != "All Actions":
            for frame_results in action_results_front:
                for track_id, action in frame_results.items():
                    if action == selected_action:
                        filtered_data.append(track_id)

            for frame_results in action_results_center:
                for track_id, action in frame_results.items():
                    if action == selected_action:
                        filtered_data.append(track_id)

        if not filtered_data:
            print(f"[DEBUG] No data found for action: {selected_action}, displaying unfiltered heatmap.")
            frame = self.generate_heatmap(self.human_detect_results_front + self.human_detect_results_center)
        else:
            frame = self.generate_heatmap_based_on_action(frame, selected_action)

        # Store the latest heatmap frame
        self.main_window.heatmap_frame = frame.copy()

        # Convert frame to QImage for displaying in QLabel
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



    def generate_heatmap_based_on_action(self, frame, selected_action):
        """Generates a heatmap based on the selected action."""
        
        filtered_data = []

        # Extract track IDs matching the selected action
        matching_track_ids = set()

        if selected_action != "All Actions":
            for frame_results in self.action_results_list_front:
                for track_id, action in frame_results.items():
                    if action.lower() == selected_action.lower():
                        matching_track_ids.add(track_id)

            for frame_results in self.action_results_list_center:
                for track_id, action in frame_results.items():
                    if action.lower() == selected_action.lower():
                        matching_track_ids.add(track_id)

        # Filter human detection results based on matching track IDs
        for detection in self.human_detect_results_front + self.human_detect_results_center:
            track_id = detection.get("track_id")
            if selected_action == "All Actions" or track_id in matching_track_ids:
                filtered_data.append(detection)

        print(f"[DEBUG] Filtered Data for action '{selected_action}':", filtered_data)

        if not filtered_data:
            print("[DEBUG] No data matched the selected action.")
            return frame  # Return original frame if no matches

        return self.create_heatmap_from_data(filtered_data)






    def generate_full_heatmap(self):
        """Generates the full heatmap and stores it in heatmap_frame."""
        heatmap = np.zeros((480, 640, 3), dtype=np.uint8)  # Replace with actual heatmap logic
        self.main_window.heatmap_frame = heatmap  # Store the generated heatmap
        return heatmap

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

    def create_heatmap_from_data(self, filtered_data):
        """Generates a heatmap image from filtered detection data."""
        if not filtered_data:
            print("[DEBUG] No data available for heatmap generation.")
            return np.zeros((480, 640, 3), dtype=np.uint8)  # Blank heatmap

        heatmap = np.zeros((480, 640), dtype=np.float32)

        for person in filtered_data:
            bbox = person.get("bbox")
            if bbox:
                x, y, _, _ = bbox
                x = max(0, min(x, 639))
                y = max(0, min(y, 479))
                heatmap[y, x] += 1  # Increase intensity at detected locations

        if heatmap.max() > 0:
            heatmap = np.uint8(255 * (heatmap / heatmap.max()))
        else:
            heatmap = np.uint8(heatmap)

        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap_colored

    
    def get_latest_heatmap_frame(self):
        """Returns the most recent heatmap frame if available."""
        if hasattr(self.main_window, "heatmap_frame") and self.main_window.heatmap_frame is not None:
            return self.main_window.heatmap_frame
        else:
            print("[DEBUG] No heatmap frame available, generating a new one.")
            return self.generate_full_heatmap()
        
    def generate_heatmap(self, detection_results):
        """Generates a heatmap from detected human positions."""
        heatmap = np.zeros((480, 640), dtype=np.float32)  # Create a blank heatmap

        for detection in detection_results:
            if isinstance(detection, dict) and "bbox" in detection:
                x, y, _, _ = detection["bbox"]
                if 0 <= x < 640 and 0 <= y < 480:
                    heatmap[y, x] += 1  # Increase intensity at detected positions

        # Normalize and apply colormap
        if heatmap.max() > 0:
            heatmap = np.uint8(255 * (heatmap / heatmap.max()))
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color mapping
        else:
            heatmap_colored = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image if no detections

        return heatmap_colored
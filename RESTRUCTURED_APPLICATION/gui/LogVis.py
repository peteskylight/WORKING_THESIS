from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QPushButton, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import numpy as np

class LogsTab(QWidget):
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Action Recognition Logs")
        
        self.layout = QVBoxLayout()
        
        # Create a QGraphicsView and Scene for the log table
        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        
        # Create a table widget inside QGraphicsView
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(3)
        self.log_table.setHorizontalHeaderLabels(["Person ID", "Action", "Timestamp (s)"])
        self.log_table.setEditTriggers(QTableWidget.NoEditTriggers)  # Make table read-only
        
        self.scene.addWidget(self.log_table)
        self.layout.addWidget(self.graphics_view)
        
        # Camera preview labels
        self.center_video_preview_label_2 = QLabel("Center Camera Preview")
        self.front_video_preview_label_2 = QLabel("Front Camera Preview")
        
        self.center_video_preview_label_2.setAlignment(Qt.AlignCenter)
        self.front_video_preview_label_2.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.center_video_preview_label_2)
        self.layout.addWidget(self.front_video_preview_label_2)
        
        # Play/Pause button for logs
        self.play_pause_button_analytics_2 = QPushButton("Play")
        self.play_pause_button_analytics_2.clicked.connect(self.toggle_play_pause_logs)
        self.layout.addWidget(self.play_pause_button_analytics_2)
        
        self.setLayout(self.layout)
        
        # Variables for controlling log updates
        self.is_playing = False
        self.log_update_interval = 1  # Placeholder for future update frequency adjustments
    
    def update_logs(self, action_results_list, camera_source):
        """
        Updates the log table with new entries from detected actions.
        :param action_results_list: List of dictionaries containing track ID and detected action per frame.
        :param camera_source: String indicating which camera (Front/Center) the actions came from.
        """
        self.log_table.setRowCount(0)  # Clear previous logs
        
        fps = 18  # Video frame rate
        
        for frame_idx, action_dict in enumerate(action_results_list):
            timestamp = frame_idx / fps  # Convert frame index to seconds
            for person_id, action in action_dict.items():
                row_position = self.log_table.rowCount()
                self.log_table.insertRow(row_position)
                self.log_table.setItem(row_position, 0, QTableWidgetItem(f"Person {person_id}"))
                self.log_table.setItem(row_position, 1, QTableWidgetItem(action))
                self.log_table.setItem(row_position, 2, QTableWidgetItem(f"{timestamp:.2f} s"))
    
    def clear_logs(self):
        """Clears all logs from the table."""
        self.log_table.setRowCount(0)
    
    def toggle_play_pause_logs(self):
        """Toggles play/pause for logs."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_pause_button_analytics_2.setText("Pause")
            # Future implementation: Start updating logs based on video playback
        else:
            self.play_pause_button_analytics_2.setText("Play")
            # Future implementation: Pause log updates

    def display_video_frame(self, label, frame):
        """ Converts OpenCV frame to QPixmap and updates QLabel """
        if frame is not None:
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
            label.setPixmap(QPixmap.fromImage(q_img))

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
            scaled_pixmap = pixmap.scaled(self.main_window.center_video_preview_label_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.main_window.center_video_preview_label_2.setPixmap(scaled_pixmap)


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
            scaled_pixmap = pixmap.scaled(self.main_window.front_video_preview_label_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.main_window.front_video_preview_label_2.setPixmap(scaled_pixmap)

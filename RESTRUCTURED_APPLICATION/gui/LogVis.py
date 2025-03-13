from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, 
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSizePolicy, 
    QGraphicsProxyWidget
)
from PySide6.QtCharts import QChartView, QChart  
from PySide6.QtGui import QPainter, QPixmap, QImage  
from PySide6.QtCore import Qt, QTimer
import numpy as np

class LogsTab(QWidget):
    def __init__(self, main_window, action_results_list_front, action_results_list_center):
        super().__init__()
        self.action_results_list_front = action_results_list_front
        self.action_results_list_center = action_results_list_center
        self.main_window = main_window

        # Track last processed frame index for incremental updates
        self.last_processed_frame_front = -1  
        self.last_processed_frame_center = -1  

        # Timer setup for periodic log updates
        self.is_playing = False
        self.log_update_interval = 1000  # Update every 1 second
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_logs_periodically)
        self.setWindowTitle("Action Recognition Logs")

        # Create Graphics Scene
        scene = QGraphicsScene()

       # Create a widget container for layout management
        container_widget = QWidget()
        layout = QVBoxLayout()
        container_widget.setLayout(layout)

        # Log table setup
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(3)
        self.log_table.setHorizontalHeaderLabels(["Person ID", "Action", "Timestamp"])
        self.log_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.log_table)

        # Set container size to match QGraphicsView dimensions
        container_widget.setMinimumSize(461, 191)  
        container_widget.setMaximumSize(461, 191)  

        # Add widget inside a QGraphicsProxyWidget
        table_proxy = QGraphicsProxyWidget()
        table_proxy.setWidget(container_widget)
        table_proxy.setPos(0, 0)  # Align to top-left corner

        # Add to scene
        scene.addItem(table_proxy)

        # Ensure QGraphicsView uses this scene
        self.main_window.DataChart_2.setScene(scene)
        self.main_window.DataChart_2.setRenderHint(QPainter.RenderHint.Antialiasing)


        # Camera preview labels
        self.center_video_preview_label_2 = self.main_window.findChild(QLabel, "center_video_preview_label_2")
        self.front_video_preview_label_2 = self.main_window.findChild(QLabel, "front_video_preview_label_2")
        self.play_pause_button_analytics_2 = self.main_window.findChild(QPushButton, "play_pause_button_analytics_2")

        # Play/Pause button for logs
        self.play_pause_button_analytics_2.clicked.connect(self.toggle_play_pause_logs)

        # Variables for controlling log updates
        self.is_playing = False
        self.log_update_interval = 1000  # 1 second interval for updating logs
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_logs_periodically)

    def toggle_play_pause_logs(self):
        """Toggle play/pause for log updates."""
        self.is_playing = not self.is_playing
        self.play_pause_button_analytics_2.setText("Pause" if self.is_playing else "Play")
        
        if self.is_playing:
            self.timer.start(self.log_update_interval)  # Start updating logs
        else:
            self.timer.stop()  # Stop updating logs

    def update_logs_from_list(self, logs):
        """Update the log table with new log entries."""
        self.log_table.setRowCount(len(logs))
        for row, (person_id, action, timestamp) in enumerate(logs):
            self.log_table.setItem(row, 0, QTableWidgetItem(str(person_id)))
            self.log_table.setItem(row, 1, QTableWidgetItem(action))
            self.log_table.setItem(row, 2, QTableWidgetItem(str(timestamp)))

    def update_logs_periodically(self):
        """Fetch new logs every second and append them in descending order (newest first)."""
        if not self.is_playing or not self.main_window:
            return  

        new_logs = []
        fps = 18  # Video frame rate

        # Process new frames from front camera
        while self.last_processed_frame_front + 1 < len(self.action_results_list_front):
            self.last_processed_frame_front += 1
            frame_idx = self.last_processed_frame_front
            timestamp = frame_idx / fps  # Correct timestamp calculation
            action_dict = self.action_results_list_front[frame_idx]
            for person_id, action in action_dict.items():
                new_logs.append((person_id, action, f"{timestamp:.2f} s"))

        # Process new frames from center camera
        while self.last_processed_frame_center + 1 < len(self.action_results_list_center):
            self.last_processed_frame_center += 1
            frame_idx = self.last_processed_frame_center
            timestamp = frame_idx / fps
            action_dict = self.action_results_list_center[frame_idx]
            for person_id, action in action_dict.items():
                new_logs.append((person_id, action, f"{timestamp:.2f} s"))

        # Append new logs and keep newest at the top
        self.append_logs(new_logs)

    def append_logs(self, new_logs):
        """Insert new logs at the top so the newest logs appear first."""
        for person_id, action, timestamp in reversed(new_logs):  # Reverse order (newest first)
            self.log_table.insertRow(0)  # Insert at the top
            self.log_table.setItem(0, 0, QTableWidgetItem(f"Person {person_id}"))
            self.log_table.setItem(0, 1, QTableWidgetItem(action))
            self.log_table.setItem(0, 2, QTableWidgetItem(timestamp))

    
    def update_logs(self, action_results_list ):
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

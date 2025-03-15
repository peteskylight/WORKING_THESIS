from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, 
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSizePolicy, 
    QGraphicsProxyWidget, QSlider
)
from PySide6.QtCharts import QChartView, QChart  
from PySide6.QtGui import QPainter, QPixmap, QImage, QColor
from PySide6.QtCore import Qt, QTimer, Signal
import numpy as np

class LogsTab(QWidget):
    row_selected = Signal(int)  # Signal to notify video player of timestamp

    def __init__(self, main_window, action_results_list_front, action_results_list_center, min_time, max_time):
        super().__init__()
        self.action_results_list_front = action_results_list_front
        self.action_results_list_center = action_results_list_center
        self.main_window = main_window
        self.min_time = min_time
        self.max_time = max_time
        self.min_frame = 0
        self.max_frame = 0
        self.main_window = main_window  # Store reference to main window
        
      
        self.TimeLabel = self.main_window.findChild(QLabel, "TimeLabel")

        # If QLabel doesn't exist, create one and add it
        if not self.TimeLabel:
            self.TimeLabel = QLabel("Time Range: 0.00s - 0.00s")
            if not self.layout():
                layout = QVBoxLayout()
                self.setLayout(layout)
            else:
                layout = self.layout()

            layout.addWidget(self.TimeLabel)
            self.TimeLabel.setFixedSize(200, 30)  # Adjust height if needed

        # Connect slider movement to update time label
        self.main_window.timeFrameRangeSlider_2.valueChanged.connect(self.update_time_label)
                

        # Track last processed frame index for incremental updates
        self.last_processed_frame_front = -1  
        self.last_processed_frame_center = -1  

        # Timer setup for periodic log updates
        self.is_playing = False
        self.log_update_interval = 500  # 500ms interval (0.5 seconds)
        self.log_update_timer = QTimer()
        self.log_update_timer.timeout.connect(self.update_logs_periodically)
        self.log_update_timer.start(self.log_update_interval)
        self.setWindowTitle("Action Recognition Logs")

        # Log table setup
        self.log_table = self.main_window.findChild(QTableWidget, "tableWidget")
        self.log_table.setColumnCount(3)
        self.log_table.setHorizontalHeaderLabels(["Person ID", "Action", "Timestamp"])
        self.log_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Camera preview labels
        self.center_video_preview_label_2 = self.main_window.findChild(QLabel, "center_video_preview_label_2")
        self.front_video_preview_label_2 = self.main_window.findChild(QLabel, "front_video_preview_label_2")
        self.play_pause_button_analytics_2 = self.main_window.findChild(QPushButton, "play_pause_button_analytics_2")

        # Play/Pause button for logs
        self.play_pause_button_analytics_2.clicked.connect(self.toggle_play_pause_logs)
        self.main_window.timeFrameRangeSlider_2.valueChanged.connect(self.update_logs_periodically)
        self.log_table.cellDoubleClicked.connect(self.on_row_double_clicked)
        self.row_selected.connect(self.main_window.update_video_position)

    def toggle_play_pause_logs(self):
        """Toggle play/pause for log updates."""
        self.is_playing = not self.is_playing
        self.play_pause_button_analytics_2.setText("Pause" if self.is_playing else "Play")
        
        if self.is_playing:
            self.log_update_timer.start(self.log_update_interval)  # Use self.log_update_interval
        else:
            self.log_update_timer.stop()

    def update_logs_from_list(self, logs):
        """Update the log table with new log entries."""
        self.log_table.setRowCount(len(logs))
        for row, (person_id, action, timestamp) in enumerate(logs):
            self.log_table.setItem(row, 0, QTableWidgetItem(str(person_id)))
            self.log_table.setItem(row, 1, QTableWidgetItem(action))
            self.log_table.setItem(row, 2, QTableWidgetItem(str(timestamp)))

    def update_logs_periodically(self):
        """Fetch new logs every 0.5 seconds and update dynamically while filtering logs within the selected time range."""
        if not self.is_playing or not self.main_window:
            return  
        # Get the new time range from the slider

        value = self.main_window.timeFrameRangeSlider_2.value()
        if isinstance(value, tuple):
            min_time, max_time = value
        else:
            min_time, max_time = value, value 
        
        fps = 20  # Assuming 20 FPS
        self.min_time = min_time / fps  # Convert frames to seconds
        self.max_time = max_time / fps 
        new_logs = []
        # Reset processing index since we want to re-evaluate logs in the new time range
        self.last_processed_frame_front = -1
        self.last_processed_frame_center = -1

        # Process front camera frames
        for frame_idx in range(len(self.action_results_list_front)):
            timestamp = frame_idx / fps  # Convert frame index to time
            current_time = self.get_current_video_time()  # Get current playback time
            if timestamp < self.min_time or timestamp > current_time:
                continue 
            action_dict = self.action_results_list_front[frame_idx]
            for person_id, action in action_dict.items():
                if action and action.lower() != "no action":  # Filter out "No Action"
                    new_logs.append((person_id, action, f"({timestamp:.2f})"))

        # Process center camera frames
        for frame_idx in range(len(self.action_results_list_center)):
            timestamp = frame_idx / fps
            current_time = self.get_current_video_time()  # Get current playback time
            if timestamp < self.min_time or timestamp > current_time:
                continue 
            action_dict = self.action_results_list_center[frame_idx]
            for person_id, action in action_dict.items():
                if action and action.lower() != "no action":  # Filter out "No Action"
                    new_logs.append((person_id, action, f"({timestamp:.2f})"))

        # Sort logs by timestamp (newest first)
        new_logs.sort(reverse=True, key=lambda log: float(log[2][1:-1]))

        # Update table dynamically
        self.update_logs_from_list(new_logs)

    def append_logs(self, new_logs):
        """Insert new logs at the top so the newest logs appear first."""
        for person_id, action, timestamp in reversed(new_logs):  # Reverse order (newest first)
            self.log_table.insertRow(0)  # Insert at the top
            self.log_table.setItem(0, 0, QTableWidgetItem(f"Person {person_id}"))
            self.log_table.setItem(0, 1, QTableWidgetItem(action))
            self.log_table.setItem(0, 2, QTableWidgetItem(timestamp))


    def display_video_frame(self, label, frame):
        """ Converts OpenCV frame to QPixmap and updates QLabel """
        if frame is not None:
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
            label.setPixmap(QPixmap.fromImage(q_img))

    def update_logs(self, action_results_list ):
            """
            Updates the log table with new entries from detected actions.
            :param action_results_list: List of dictionaries containing track ID and detected action per frame.
            :param camera_source: String indicating which camera (Front/Center) the actions came from.
            """
            self.log_table.setRowCount(0)  # Clear previous logs
            
            fps = 20  # Video frame rate
            
            for frame_idx, action_dict in enumerate(action_results_list):
                timestamp = frame_idx / fps  # Convert frame index to seconds
                for person_id, action in action_dict.items():
                    row_position = self.log_table.rowCount()
                    self.log_table.insertRow(row_position)
                    self.log_table.setItem(row_position, 0, QTableWidgetItem(f"Person {person_id}"))
                    self.log_table.setItem(row_position, 1, QTableWidgetItem(action))
                    self.log_table.setItem(row_position, 2, QTableWidgetItem(f"{timestamp:.2f} s"))

    def get_current_video_time(self):
        """Estimate the current playback time using the time range slider."""
        current_frame = self.main_window.timeFrameRangeSlider_2.value()[1]  # Get max slider value (assuming it's in frames)
        fps = 20  # Adjust based on actual FPS
        return current_frame / fps  # Convert frames to seconds



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

    def update_time_label(self):
        """Update QLabel with the current time range from the slider."""
        if self.main_window.timeFrameRangeSlider_2:
            min_time, max_time = self.main_window.timeFrameRangeSlider_2.value()  # Get slider values
            fps = 20  # Adjust based on actual FPS
            min_time_sec = min_time / fps
            max_time_sec = max_time / fps
            self.TimeLabel.setText(f"Time Range: {min_time_sec:.2f}s - {max_time_sec:.2f}s")
            self.TimeLabel.repaint()

    def on_row_double_clicked(self, row, column):
        timestamp_item = self.log_table.item(row, 2)  # Assuming timestamp is in the 3rd column
        if timestamp_item:
            try:
                raw_timestamp = timestamp_item.text().strip("()")  # Remove parentheses
                timestamp_seconds = float(raw_timestamp)  # Convert to float (seconds)
                
                print(f"Double-clicked row {row}, emitting timestamp {timestamp_seconds} seconds")

                self.row_selected.emit(timestamp_seconds)  # Emit in seconds
                self.highlight_row(row)
            except ValueError:
                print(f"Invalid timestamp format: {timestamp_item.text()}")  # Debugging




    def highlight_row(self, row):
        for r in range(self.log_table.rowCount()):
            for c in range(self.log_table.columnCount()):
                item = self.log_table.item(r, c)
                if item:
                    item.setBackground(Qt.white)  # Reset all rows
        
        for c in range(self.log_table.columnCount()):
            item = self.log_table.item(row, c)
            if item:
                item.setBackground(Qt.yellow)  # Highlight selecte


            



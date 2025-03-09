from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QPushButton, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage

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

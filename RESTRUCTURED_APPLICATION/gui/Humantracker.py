from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel
from PySide6.QtCore import Qt


class Human_Tracker(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.grid_size = (7, 6)  # 7 columns, 6 rows
        
        # Layout setup
        self.layout = QVBoxLayout()
        self.grid_layout = QGridLayout()
        
        # Create a 7x6 grid with labels
        self.cell_labels = []
        for row in range(self.grid_size[1]):
            row_labels = []
            for col in range(self.grid_size[0]):
                label = QLabel("0")  # Default to 0 detections
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setStyleSheet("border: 1px solid black; min-width: 40px; min-height: 40px;")
                self.grid_layout.addWidget(label, row, col)
                row_labels.append(label)
            self.cell_labels.append(row_labels)
        
        self.layout.addLayout(self.grid_layout)
        self.setLayout(self.layout)
        
    def update_grid(self, human_detect_results_front, human_detect_results_center):
        # Reset all counts
        grid_counts = [[0 for _ in range(self.grid_size[0])] for _ in range(self.grid_size[1])]
        
        # Process detections from both cameras
        for detections in (human_detect_results_front, human_detect_results_center):
            if detections is not None:
                for position in detections.values():
                    col, row = position  # Assuming (x, y) format
                    if 0 <= row < self.grid_size[1] and 0 <= col < self.grid_size[0]:
                        grid_counts[row][col] += 1
        
        # Update labels with new counts
        for row in range(self.grid_size[1]):
            for col in range(self.grid_size[0]):
                self.cell_labels[row][col].setText(str(grid_counts[row][col]))

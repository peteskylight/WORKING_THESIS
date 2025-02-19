import sys
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

class ActionVisualization(QWidget):
    def __init__(self, action_results_list_front, action_results_list_center):
        super().__init__()
        self.setWindowTitle("Action Analytics")
        self.resize(800, 600)

        self.action_results_list_front = action_results_list_front or []
        self.action_results_list_center = action_results_list_center or []

        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.plot_action_data()

    def plot_action_data(self):
        """Plots action occurrences over time."""
        action_labels = ['Looking Down', 'Looking Forward', 'Looking Left', 'Looking Right', 'Looking Up']
        
        # Simulated time axis
        time_intervals = np.arange(len(self.action_results_list_front))

        action_counts = {action: np.zeros(len(time_intervals)) for action in action_labels}

        # Count actions per time step
        for t, actions in enumerate(self.action_results_list_front):
            if actions:  # Check if actions exist
                for action in actions:
                    if action in action_counts:
                        action_counts[action][t] += 1

        self.ax.clear()

        for action, counts in action_counts.items():
            self.ax.plot(time_intervals, counts, label=action)

        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Action Count")
        self.ax.set_title("Student Actions Over Time")
        self.ax.legend()

        self.canvas.draw()

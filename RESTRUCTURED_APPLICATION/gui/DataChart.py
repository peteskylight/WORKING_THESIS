from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtWidgets import QGraphicsScene, QSizePolicy, QVBoxLayout, QPushButton, QWidget,QFileDialog, QApplication
from PySide6.QtGui import QPainter, QFont
from PySide6.QtCore import Qt, QThread, Signal
import pandas as pd
import os

class ActionVisualization:
    def __init__(self, main_window, action_results_list_front, action_results_list_center, min_time, max_time):
        self.action_results_list_front = action_results_list_front
        self.action_results_list_center = action_results_list_center
        self.min_time = min_time
        self.max_time = max_time
        self.main_window = main_window
        self.action_labels = ['Extending Right Arm', 'Standing', 'Sitting']
        self.active_actions = set(self.action_labels)

        # Create chart
        self.chart = QChart()
        self.chart.setTitle("Actions Over Time")
        self.chart.setAnimationOptions(QChart.AnimationOption.AllAnimations)

        # Create series for each action
        self.series_dict = {label: QLineSeries() for label in self.action_labels}
        for label, series in self.series_dict.items():
            series.setName(label)
            self.chart.addSeries(series)

        # Configure X-axis (Time in Seconds)
        self.axis_x = QValueAxis()
        self.axis_x.setTitleText("Time (seconds)")
        self.axis_x.setLabelFormat("%.1f")
        self.axis_x.setRange(self.min_time / 30.0, self.max_time / 30.0)
        self.axis_x.setTickCount(min(10, (self.max_time - self.min_time) // 30 + 1))
        font = QFont()
        font.setPointSize(12)
        self.axis_x.setLabelsFont(font)
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        
        for series in self.series_dict.values():
            series.attachAxis(self.axis_x)

        # Configure Y-axis (Action Counts)
        self.axis_y = QValueAxis()
        self.axis_y.setTitleText("Students")
        self.axis_y.setRange(0, 40)
        self.axis_y.setTickCount(9)  # Ensures labels are evenly spaced
        self.axis_y.setLabelFormat("%d")  # Ensures integer values
        self.axis_y.setLabelsFont(font)
        font.setPointSize(8)
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
                
        for series in self.series_dict.values():
            series.attachAxis(self.axis_y)

        # Set up chart in QGraphicsView
        self.scene = QGraphicsScene()
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.chart_view.setMinimumSize(800, 600)

        self.scene.addWidget(self.chart_view)
        self.main_window.DataChart.setScene(self.scene)
        self.chart_view.setMinimumSize(self.main_window.DataChart.size())

        # Create action selection buttons
        self.button_widget = QWidget()
        self.button_layout = QVBoxLayout()
        self.buttons = {}
        for label in self.action_labels:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setChecked(True)
            button.clicked.connect(lambda checked, lbl=label: self.toggle_action(lbl, checked))
            self.button_layout.addWidget(button)
            self.buttons[label] = button
        self.button_widget.setLayout(self.button_layout)
        self.proxy_widget = self.scene.addWidget(self.button_widget)
        self.proxy_widget.setPos(10, 10)

        # Connect slider to update chart dynamically
        self.main_window.timeFrameRangeSlider.valueChanged.connect(self.update_chart)

        # Connect button to export function
        try:
            self.main_window.see_full_data_button.clicked.disconnect(self.handle_export)
        except TypeError:
            pass  # Ignore if no connection exists
        self.main_window.see_full_data_button.clicked.connect(self.handle_export)


        self.populate_chart()

    def populate_chart(self):
        for series in self.series_dict.values():
            series.clear()

        for frame_index in range(self.min_time, self.max_time + 1):
            time_value = frame_index / 30.0  # Convert frame index to seconds
            combined_counts = {label: 0 for label in self.action_labels}

            if frame_index < len(self.action_results_list_front):
                for action in self.action_results_list_front[frame_index].values():
                    if action in combined_counts:
                        combined_counts[action] += 1

            if frame_index < len(self.action_results_list_center):
                for action in self.action_results_list_center[frame_index].values():
                    if action in combined_counts:
                        combined_counts[action] += 1

            for label, series in self.series_dict.items():
                if label in self.active_actions:
                    series.append(time_value, combined_counts[label])

    def update_chart(self):
        self.min_time, self.max_time = self.main_window.timeFrameRangeSlider.value()
        self.populate_chart()

        # Reconfigure X-axis dynamically
        self.chart.removeAxis(self.axis_x)
        self.axis_x = QValueAxis()
        self.axis_x.setTitleText("Time (seconds)")
        self.axis_x.setLabelFormat("%.1f")
        self.axis_x.setRange(self.min_time / 30.0, self.max_time / 30.0)
        self.axis_x.setTickCount(min(10, (self.max_time - self.min_time) // 30 + 1))
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        
        for series in self.series_dict.values():
            series.attachAxis(self.axis_x)

        self.chart.update()

    def toggle_action(self, action, checked):
        if checked:
            self.active_actions.add(action)
        else:
            self.active_actions.discard(action)
        self.populate_chart()

    def export_to_excel(self):
        data = []
        for frame_index in range(self.min_time, self.max_time + 1):
            time_value = frame_index / 30.0  # Convert frame index to seconds
            frame_data = {"Time (s)": time_value}

            if frame_index < len(self.action_results_list_front):
                for track_id, action in self.action_results_list_front[frame_index].items():
                    frame_data[f"Front_Cam_Person_{track_id}"] = action

            if frame_index < len(self.action_results_list_center):
                for track_id, action in self.action_results_list_center[frame_index].items():
                    frame_data[f"Center_Cam_Person_{track_id}"] = action

            data.append(frame_data)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Save to Excel file
        file_path, _ = QFileDialog.getSaveFileName(None, "Save Excel File", "", "Excel Files (*.xlsx)")
        if file_path:
            df.to_excel(file_path, index=False)
        
    def handle_export_to_jpeg(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            return  # Prevent starting another thread while one is running
        self.thread = JpegExportThread(self.chart_view, self.populate_chart, self.action_labels, self.active_actions)
        self.thread.finished.connect(self.on_jpeg_export_complete)
        self.thread.start()

    def on_jpeg_export_complete(self, message):

        print(message) 
    def handle_export(self):
        self.export_to_excel()  # Call Excel export
        self.handle_export_to_jpeg()


class JpegExportThread(QThread):
    finished = Signal(str)  # Signal when export is complete
    exporting = False

    def __init__(self, chart_view, populate_chart, action_labels, active_actions):
        super().__init__()
        self.chart_view = chart_view
        self.populate_chart = populate_chart
        self.action_labels = action_labels
        self.active_actions = active_actions

    def run(self):
        try:
            output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
            os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

            original_active_actions = self.active_actions.copy()  # Save original actions

            # Export separate charts for each action
            for action in self.action_labels:
                print(f"📊 Exporting: {action}")  # Debug log
                self.active_actions.clear()
                self.active_actions.add(action)
                self.populate_chart()
                self.chart_view.update()
                QApplication.processEvents()  # Force UI update

                QThread.msleep(500)  # Small delay to allow GUI to update

                # Save individual action chart
                action_filename = f"{action.replace(' ', '_')}.jpg"
                action_path = os.path.join(output_dir, action_filename)
                pixmap = self.chart_view.grab()
                pixmap.save(action_path, "JPG")

            # Export a chart with all actions
            print("📊 Exporting: All Actions")  # Debug log
            self.active_actions.clear()
            self.active_actions.update(self.action_labels)  # Enable all actions
            self.populate_chart()
            self.chart_view.update()
            QApplication.processEvents()

            QThread.msleep(100)  # Small delay to allow GUI to update

            all_actions_path = os.path.join(output_dir, "All_Actions.jpg")
            pixmap = self.chart_view.grab()
            pixmap.save(all_actions_path, "JPG")

            # Restore original active actions
            self.active_actions.clear()
            self.active_actions.update(original_active_actions)
            self.populate_chart()
            self.chart_view.update()

            self.finished.emit(f"Charts saved in: {output_dir}")

        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")
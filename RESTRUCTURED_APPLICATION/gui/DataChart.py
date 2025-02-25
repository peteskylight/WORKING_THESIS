from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtWidgets import QGraphicsScene, QSizePolicy
from PySide6.QtGui import QPainter, QFont
from PySide6.QtCore import Qt

class ActionVisualization:
    def __init__(self, main_window, action_results_list_front, action_results_list_center, min_time, max_time):
        self.action_results_list_front = action_results_list_front
        self.action_results_list_center = action_results_list_center
        self.min_time = min_time  # Start frame
        self.max_time = max_time  # End frame
        self.main_window = main_window

        # Create chart
        self.chart = QChart()
        self.chart.setTitle("Concurrent Actions Over Time")
        self.chart.setAnimationOptions(QChart.AnimationOption.AllAnimations)

        # Create series
        self.series = QLineSeries()
        self.series.setName("Concurrent Actions")

        # Configure X-axis
        self.axis_x = QValueAxis()
        self.axis_x.setTitleText("Time (seconds)")
        self.axis_x.setLabelFormat("%.1f")

        font = QFont()
        font.setPointSize(12)
        self.axis_x.setLabelsFont(font)

        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.series.attachAxis(self.axis_x)

        # Configure Y-axis
        self.axis_y = QValueAxis()
        self.axis_y.setTitleText("Concurrent Actions")
        self.axis_y.setTickCount(8)
        self.axis_y.setLabelsFont(font)

        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(self.axis_y)

        # Add series to chart
        self.chart.addSeries(self.series)

        # Set up chart in QGraphicsView
        self.scene = QGraphicsScene()
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.chart_view.setMinimumSize(800, 500)

        self.scene.addWidget(self.chart_view)
        self.main_window.DataChart.setScene(self.scene)
        self.chart_view.setMinimumSize(self.main_window.DataChart.size())

        # Populate chart initially
        self.update_chart()

        # Connect slider to update function
        self.main_window.timeFrameRangeSlider.valueChanged.connect(self.on_slider_change)

    def update_chart(self):
        """ Updates the chart based on the current slider values. """
        self.series.clear()

        # Get updated min and max values from the slider
        self.min_time = self.main_window.timeFrameRangeSlider.value()[0]
        self.max_time = self.main_window.timeFrameRangeSlider.value()[1]

        frame_action_counts = {}

        for frame_index in range(self.min_time, self.max_time + 1):
            time_value = frame_index / 30.0  # Convert frame index to seconds
            count_front = len(self.action_results_list_front[frame_index]) if frame_index < len(self.action_results_list_front) else 0
            count_center = len(self.action_results_list_center[frame_index]) if frame_index < len(self.action_results_list_center) else 0
            total_concurrent_actions = count_front + count_center
            frame_action_counts[time_value] = total_concurrent_actions
            self.series.append(time_value, total_concurrent_actions)

        # Update axis ranges
        self.axis_x.setRange(self.min_time / 30.0, self.max_time / 30.0)
        self.axis_x.setTickCount(min(10, (self.max_time - self.min_time) // 30 + 1))

        max_y_value = max(40, max((p.y() for p in self.series.pointsVector()), default=0) + 5)
        self.axis_y.setRange(0, max_y_value)

        print("Updated Frame Action Counts:", frame_action_counts)

    def on_slider_change(self):
        """ Triggered when the slider is moved to dynamically update the chart. """
        self.update_chart()

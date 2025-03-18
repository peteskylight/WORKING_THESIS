from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, 
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSizePolicy, 
    QGraphicsProxyWidget, QSlider, QTableWidget

)
from PySide6.QtCharts import QChartView, QChart  
from PySide6.QtGui import QPainter, QPixmap, QImage, QColor, QImage, QPainter
from PySide6.QtCore import Qt, QTimer, Signal, QPoint, QRect
import numpy as np
from reportlab.lib.pagesizes import landscape, A4, letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from collections import defaultdict
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
        self.log_table_2 = self.main_window.findChild(QTableWidget, "tableWidget_2")  # Turning Around
        self.log_table_3  = self.main_window.findChild(QTableWidget, "tableWidget_3") 


        # Setup tableWidget (Sitting, Leaning on Desk)
        if self.log_table:
            self.log_table.setColumnCount(3)
            self.log_table.setHorizontalHeaderLabels(["Person ID", "Action", "Timestamp"])
            self.log_table.setSortingEnabled(True)

        # Setup tableWidget_2 (Turning Around)
        if self.log_table_2:
            self.log_table_2.setColumnCount(3)
            self.log_table_2.setHorizontalHeaderLabels(["Person ID", "Action", "Timestamp"])
            self.log_table_2.setSortingEnabled(True)

        # Setup tableWidget_3 (Standing, Extending Arm)
        if self.log_table_3:
            self.log_table_3.setColumnCount(3)
            self.log_table_3.setHorizontalHeaderLabels(["Person ID", "Action", "Timestamp"])
            self.log_table_3.setSortingEnabled(True)

        self.export_2 = self.main_window.findChild(QPushButton, "export_2")

        # Camera preview labels
        self.center_video_preview_label_2 = self.main_window.findChild(QLabel, "center_video_preview_label_2")
        self.front_video_preview_label_2 = self.main_window.findChild(QLabel, "front_video_preview_label_2")
        self.play_pause_button_analytics_2 = self.main_window.findChild(QPushButton, "play_pause_button_analytics_2")

        # Play/Pause button for logs
        self.play_pause_button_analytics_2.clicked.connect(self.toggle_play_pause_logs)
        self.main_window.timeFrameRangeSlider_2.valueChanged.connect(self.update_logs_periodically)
        self.log_table.cellDoubleClicked.connect(self.on_row_double_clicked)
        self.log_table_2.cellDoubleClicked.connect(self.on_row_double_clicked)
        self.log_table_3.cellDoubleClicked.connect(self.on_row_double_clicked)
        self.row_selected.connect(self.main_window.update_video_position)
        self.export_2.clicked.connect(self.handle_export_logs)


    def setup_table(self, table):
        """Configures the table's headers and settings."""
        if table:  # Ensure table exists before setting properties
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels(["Person ID", "Action", "Timestamp"])
        table.setRowCount(0)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table.setSortingEnabled(True)

    def toggle_play_pause_logs(self):
        """Toggle play/pause for log updates."""
        self.is_playing = not self.is_playing
        self.play_pause_button_analytics_2.setText("Pause" if self.is_playing else "Play")
        
        if self.is_playing:
            self.log_update_timer.start(self.log_update_interval)  # Use self.log_update_interval
        else:
            self.log_update_timer.stop()

    def update_logs_from_list(self, logs):
            """Update the tables dynamically based on action type."""
            # Clear previous logs
            self.log_table.setRowCount(0)
            self.log_table_2.setRowCount(0)
            self.log_table_3.setRowCount(0)

            for person_id, action, timestamp in logs:
                if action in ["Sitting", "Leaning on desk"]:
                    self.add_row(self.log_table, person_id, action, timestamp)
                elif action == "Turning Around":
                    self.add_row(self.log_table_2, person_id, action, timestamp)
                elif action in ["Standing", "Extending right arm", "Extending left arm"]:
                    self.add_row(self.log_table_3, person_id, action, timestamp)

    def add_row(self, table, person_id, action, timestamp):
        """Adds a new row to the specified table."""
        if timestamp is None:
    
            return
        if not isinstance(timestamp, (int, float)):
           
            return
      
        
        row_position = table.rowCount()
        table.insertRow(row_position)
        table.setItem(row_position, 0, QTableWidgetItem(f"Person {person_id}"))
        table.setItem(row_position, 1, QTableWidgetItem(action))
        table.setItem(row_position, 2, QTableWidgetItem(f"{timestamp:.2f} s"))

    def update_logs_periodically(self):
        """Fetch new logs every 0.5 seconds while filtering logs within the selected time range."""
        if not self.is_playing or not self.main_window:
            return  

        value = self.main_window.timeFrameRangeSlider_2.value()
        min_time, max_time = value if isinstance(value, tuple) else (value, value)
        
        fps = 20  # Assuming 20 FPS
        self.min_time = min_time / fps  # Convert frames to seconds
        self.max_time = max_time / fps  
        new_logs = []
        # Reset processing index
        self.last_processed_frame_front = -1
        self.last_processed_frame_center = -1

        # Process front camera frames
        for frame_idx in range(len(self.action_results_list_front)):
            timestamp = frame_idx / fps  
            current_time = self.get_current_video_time()
            if timestamp < self.min_time or timestamp > current_time:
                continue 
            for person_id, action in self.action_results_list_front[frame_idx].items():
                if action and action.lower() != "no action":
                    new_logs.append((person_id, action, timestamp))

        # Process center camera frames
        for frame_idx in range(len(self.action_results_list_center)):
            timestamp = frame_idx / fps
            current_time = self.get_current_video_time()
            if timestamp < self.min_time or timestamp > current_time:
                continue 
            for person_id, action in self.action_results_list_center[frame_idx].items():
                if action and action.lower() != "no action":
                    new_logs.append((person_id, action, timestamp))

        # Sort logs by timestamp (newest first)
        new_logs.sort(reverse=True, key=lambda log: log[2])

        # Update tables dynamically
        self.update_logs_from_list(new_logs)

    def append_logs(self, new_logs):
        """Insert new logs at the top so the newest logs appear first."""
        for person_id, action, timestamp in reversed(new_logs):
            if action in ["Sitting", "Leaning on desk"]:
                self.add_row(self.log_table, person_id, action, timestamp)
            elif action == "Turning Around":
                self.add_row(self.log_table_2, person_id, action, timestamp)
            elif action in ["Standing", "Extending right arm", "Extending left arm"]:
                self.add_row(self.log_table_3, person_id, action, timestamp)

    def update_logs(self, action_results_list):
        """Updates tables with new entries from detected actions."""
        self.log_table.setRowCount(0)
        self.log_table_2.setRowCount(0)
        self.log_table_3.setRowCount(0)

        fps = 20  # Video frame rate
        
        for frame_idx, action_dict in enumerate(action_results_list):
            timestamp = frame_idx / fps  
            for person_id, action in action_dict.items():
                if action in ["Sitting", "Leaning on desk"]:
                    self.add_row(self.log_table, person_id, action, timestamp)
                elif action == "Turning Around":
                    self.add_row(self.log_table_2, person_id, action, timestamp)
                elif action in ["Standing", "Extending right arm", "Extending left arm"]:
                    self.add_row(self.log_table_3, person_id, action, timestamp)


    def display_video_frame(self, label, frame):
        """ Converts OpenCV frame to QPixmap and updates QLabel """
        if frame is not None:
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
            label.setPixmap(QPixmap.fromImage(q_img))

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
            fps = 18  # Adjust based on actual FPS ADJUST HERE PETER 
            self.min_time_sec = min_time / fps
            self.max_time_sec = max_time / fps
            self.TimeLabel.setText(f"Time Range: {self.min_time_sec:.2f}s - {self.max_time_sec:.2f}s")
            self.TimeLabel.repaint()
            

    def on_row_double_clicked(self, row, column):
        sender_table = self.sender()  # Identify which table triggered the event

        if sender_table == self.log_table:
            timestamp_item = self.log_table.item(row, 2)
        elif sender_table == self.log_table_2:
            timestamp_item = self.log_table_2.item(row, 2)
        elif sender_table == self.log_table_3:
            timestamp_item = self.log_table_3.item(row, 2)
        else:
            print("Unknown table clicked")
            return  # Exit if sender table is not recognized

        if timestamp_item:
            try:
                raw_timestamp = timestamp_item.text().strip("()s ") 
                timestamp_seconds = float(raw_timestamp)  # Convert to float
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
                item.setBackground(Qt.yellow)  # Highlight selected
                
    def export_logs_to_pdf(self, tables, filename="logs_export.pdf"):
        """
        Exports log tables to a PDF file with raw detection times and sorted order.
        """
        if not tables:
            print("No data to export.")
            return

        doc = SimpleDocTemplate(filename, pagesize=letter)  # Portrait mode
        elements = []
        styles = getSampleStyleSheet()

        # Add Title
        elements.append(Paragraph("Log Summary Report", styles['Title']))
        elements.append(Spacer(1, 12))

        summary_dict = defaultdict(lambda: {"instances": 0, "start_time": float("inf"), "end_time": float("-inf")})

        table_titles = [
            "Sitting and Leaning on Desk",
            "Turning Around",
            "Standing, Extending Arm"
        ]

        for table, title in zip(tables, table_titles):
            if table.rowCount() == 0:
                continue
            elements.append(Paragraph("To be edited paragraph cause we plan to add something soon", styles['Normal']))
            elements.append(Paragraph(title, styles['Heading2']))

            headers = ["Person ID", "Action", "Raw Detection Time (s)"]
            raw_data = []

            for row in range(table.rowCount()):
                person_id = table.item(row, 0).text().strip() if table.item(row, 0) else "Unknown"
                action = table.item(row, 1).text().strip() if table.item(row, 1) else "Unknown"
                duration_text = table.item(row, 2).text().strip() if table.item(row, 2) else "0.0"

                try:
                    start_time = float(duration_text.replace("s", "").strip())  # Use raw detection time
                    raw_data.append([person_id, action, f"{start_time:.2f}s"])

                    # Update summary dictionary
                    key = (person_id, action)
                    summary_dict[key]["instances"] += 1
                    summary_dict[key]["start_time"] = min(summary_dict[key]["start_time"], start_time)
                    summary_dict[key]["end_time"] = max(summary_dict[key]["end_time"], start_time)

                except ValueError:
                    print(f"Warning: Invalid duration '{duration_text}' for {person_id}, {action}. Skipping.")
                    continue  # Skip invalid entries

            # Sort by **Time First**, then **Person ID**
            raw_data.sort(key=lambda x: (float(x[2].replace("s", "")), x[0]))

            # Create a Table for Each Section
            section_table = Table([headers] + raw_data, colWidths=[80, 120, 150])
            section_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
            ]))

            elements.append(section_table)
            elements.append(Spacer(1, 18))

        # Convert summary dictionary to sorted list
        summary_headers = ["Person ID", "Action", "Instances", "Timestamp (s)"]
        formatted_logs = []

        for (person, action), times in summary_dict.items():
            if times["start_time"] == float("inf") or times["end_time"] == float("-inf"):
                continue  # Skip invalid entries

            timestamp = f"{times['start_time']:.2f}s - {times['end_time']:.2f}s"
            formatted_logs.append([person, action, str(times["instances"]), timestamp])

        # **Sort summary table by Person ID first, then by Time**
        formatted_logs.sort(key=lambda x: (x[0], float(x[3].split("s")[0])))

        # Create Summary Table
        summary_table = Table([summary_headers] + formatted_logs, colWidths=[80, 120, 80, 150])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
        ]))
        elements.insert(2, Paragraph("To be edited paragraph cause we plan to add something soon", styles['Normal']))
        elements.insert(3, summary_table)  # Insert summary before log tables
        elements.insert(4, Spacer(1, 18))

        doc.build(elements)
        print(f"PDF saved as {filename}")
    def handle_export_logs(self):
        tables = [self.log_table, self.log_table_2, self.log_table_3]
        tables = [table for table in tables if table]  # Remove None tables

        action_timeline = defaultdict(lambda: {"first": None, "last": None})

        for table in tables:
            if table.rowCount() == 0:
                continue  # Skip empty tables
            
            for row in range(table.rowCount()):
                person = table.item(row, 0).text().strip() if table.item(row, 0) else "Unknown Person"
                action = table.item(row, 1).text().strip() if table.item(row, 1) else "Unknown Action"
                start_time_text = table.item(row, 2).text().strip() if table.item(row, 2) else "0"

                # Convert to float
                try:
                    start_time = float(start_time_text.replace("s", "").strip())
                except ValueError:
                    continue  # Skip invalid time values

                key = (person, action)

                # Ensure first time is set properly
                if action_timeline[key]["first"] is None or start_time < action_timeline[key]["first"]:
                    action_timeline[key]["first"] = start_time
                
                # Always update last occurrence (ensuring it's greater than first)
                if action_timeline[key]["last"] is None or start_time > action_timeline[key]["last"]:
                    action_timeline[key]["last"] = start_time

        # Sort actions by extracted number
        def extract_person_number(person_id):
            digits = "".join(filter(str.isdigit, person_id))
            return int(digits) if digits else float("inf")

        sorted_timeline = sorted(action_timeline.items(), key=lambda x: extract_person_number(x[0][0]))

        summary_data = []

        # Compute time ranges
        for (person, action), times in sorted_timeline:
            first_time = times["first"]
            last_time = times["last"]
            if first_time is not None and last_time is not None:
                summary_data.append([person, action, f"{first_time:.2f} s → {last_time:.2f} s"])

                # Debugging output
                print(f"✅ {person} - {action} | Start: {first_time} s | End: {last_time} s")

        # Export logs with new summary format
        self.export_logs_to_pdf(tables, "logs_export.pdf")
        print(summary_data)
       

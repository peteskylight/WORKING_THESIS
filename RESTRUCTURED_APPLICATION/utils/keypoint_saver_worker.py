import numpy as np
from PySide6.QtCore import QObject, Slot

class KeypointSaverWorker(QObject):
    def __init__(self):
        super().__init__()
        self.queue = []
        self.running = True

    @Slot()
    def process(self):
        while self.running:
            if self.queue:
                keypoints, path = self.queue.pop(0)
                np.save(path, keypoints)

    def add_to_queue(self, keypoints, path):
        self.queue.append((keypoints, path))

    def stop(self):
        self.running = False

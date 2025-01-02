import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator

class DrawingUtils:
    def __init__(self) -> None:
        pass

    def draw_bounding_box(self, frame, box):
        # Draw bounding box
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, "Tester", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def drawPoseLandmarks(self, cropped_image, normalized_keypoints): #Requires cropped image
        for keypointsResults in normalized_keypoints:
            x = keypointsResults[0]
            y = keypointsResults[1]
            #print("X: {} | Y: {}".format(x,y))
            cv2.circle(cropped_image, (int(x * cropped_image.shape[1]), int(y * cropped_image.shape[0])),
                                    3, (0, 255, 0), -1)



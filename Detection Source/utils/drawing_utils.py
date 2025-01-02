import cv2
import numpy as np

class DrawingUtils:
    def __init__(self) -> None:
        pass
    
    def draw_bboxes(self, video_frames, detections):
        output_video_frames = []

        for frame, student_dict in zip(video_frames, detections):
            #print(type(student_dict))  # Add this line to check the type
            for track_id, bbox in student_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Student ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_video_frames.append(frame)

        return output_video_frames

    def drawLandmarks(self, frames, keypoints_normalized):
        for frame, detections in zip(frames, keypoints_normalized):
            #flattenedKeypoints = keypoints_normalized.flatten()
            for keypointsResults in detections:
                x = keypointsResults[0]
                y = keypointsResults[1]
                cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])),
                                        3, (0, 255, 0), -1)
            return frame

    def draw_room_bbox(self, video_frames, detections):
        output_video_frames1 = []

        for frame, boxes in zip(video_frames, detections):
            for box in boxes:
                x1, y1, x2, y2 = map(int, box['box'])
                class_name = box['class']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Class: {class_name}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # cv2.imshow("MONITOR", frame)
            # cv2.waitKey(10)
            output_video_frames1.append(frame)
        
        return output_video_frames1 


    def draw_keypoints(self,video_frames, pose_results, detections):
        output_video_keypoints = []
        
        for frame, keypoints_dict, student_dict in zip(video_frames, pose_results, detections):
            for track_id in keypoints_dict:
                if track_id in student_dict:
                    keypoints = keypoints_dict[track_id]
                    bbox = student_dict[track_id]
                    bbox_x, bbox_y, bbox_w, bbox_h = bbox

                    # Crop the frame using the bounding box coordinates
                    cropped_frame = frame[int(bbox_y):int(bbox_h), int(bbox_x):int(bbox_w)]

                    for keypoint in keypoints:
                        x = int(keypoint[0] * cropped_frame.shape[1])
                        y = int(keypoint[1] * cropped_frame.shape[0])
                        cv2.circle(cropped_frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
                    
                    # Place the cropped frame back into the original frame
                    frame[int(bbox_y):int(bbox_h), int(bbox_x):int(bbox_w)] = cropped_frame
        
            output_video_keypoints.append(frame)
                
        return output_video_keypoints


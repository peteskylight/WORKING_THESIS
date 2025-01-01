import os

from ultralytics import YOLO

from trackers import StudentTracker
from utils import(VideoUtils,
                  DrawingUtils,
                  Tools)

def main():

    #===INITIALIZATIONS===
    student_tracker = StudentTracker(humanDetectionModel='yolov8n.pt',
                                     humanDetectConf=0.4,
                                     humanPoseModel='yolov8n-pose.pt',
                                     humanPoseConf=0.4
                                     )

    video_utils = VideoUtils()
    drawing_utils = DrawingUtils()
    tools_utils = Tools()
    
    #DIRECTORIES===
    
    input_video_path = Tools.convert_slashes(r"C:\Users\peter\Desktop\WORKING THESIS FILES\Detection Source\test_videos\HD\Center.mp4")
    output_video_orig_path = Tools.getAbsPath("Detection Source/output_videos/TESTORIG5.avi")
    output_video_white_path = Tools.getAbsPath("Detection Source/output_videos/TESTWHITE5.avi")
    student_detections_results = Tools.convert_slashes(r"C:\Users\peter\Desktop\WORKING THESIS FILES\Detection Source\cache\student_detections_results.pkl")
    
    #READ VIDEO===
    video_frames = video_utils.read_video(video_path=input_video_path,
                                          resize_frames=False)
    
    #HEIGHT & WIDTH
    
    video_height = video_frames[0].shape[0]
    video_width = video_frames[0].shape[1]
    
    #===STUDENT DETECTION=== 
    student_detections = student_tracker.detect_frames(frames=video_frames,
                                                       read_from_stub=None,
                                                       stub_path=student_detections_results
                                                       )
    
    #===POSE ESTIMATION===
    student_pose_results = student_tracker.detect_keypoints(frames=video_frames,
                                                            student_dicts=student_detections)
    
    
    #Generate White Frames
    white_frames = []
    for frame in video_frames:
        generate = video_utils.generate_white_frame(height=video_height,
                                                        width=video_width)
        white_frames.append(generate)
    
    
    
    #DRAW OUTPUTS FOR WHITEFRAME
    output_white_frames = drawing_utils.draw_bboxes(video_frames = white_frames,
                                                    detections=student_detections,
                                                    )
    
    output_white_frames = drawing_utils.draw_keypoints(video_frames=output_white_frames,
                                                       pose_results=student_pose_results,
                                                       detections=student_detections,
                                                       )
    
    
    #===DRAW OUTPUTS IN ORIGINAL FRAME
    output_video_frames = drawing_utils.draw_bboxes(video_frames=video_frames,
                                                    detections=student_detections
                                                    )
    
    output_video_frames = drawing_utils.draw_keypoints(video_frames=output_video_frames,
                                                       pose_results=student_pose_results,
                                                       detections=student_detections,
                                                       )
    
    #EXPORT ORIGINAL VIDEO
    video_utils.save_video(output_video_frames=output_video_frames,
               output_video_path = output_video_orig_path,
               monitorFrames = False #Change me when everything is fuckedup
               )
    
    #EXPORT WHITE FRAMES
    video_utils.save_video(output_video_frames=output_white_frames,
               output_video_path=output_video_white_path,
               monitorFrames=False #Change me when everything is fuckedup
               )
        
if __name__ == "__main__":
    main()
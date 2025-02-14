import cv2
import moviepy.editor as mp
import os

def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def trim_video(input_path, output_path, target_frames):
    clip = mp.VideoFileClip(input_path)
    fps = clip.fps
    duration = target_frames / fps  # New duration to match target frames
    trimmed_clip = clip.subclip(0, duration)
    trimmed_clip.write_videofile(output_path, codec='libx264', fps=fps)

def main(video1_path, video2_path, output1_path, output2_path):
    frames1 = get_frame_count(video1_path)
    frames2 = get_frame_count(video2_path)
    
    min_frames = min(frames1, frames2)  # Get the lower frame count
    
    if frames1 > min_frames:
        trim_video(video1_path, output1_path, min_frames)
    else:
        os.rename(video1_path, output1_path)  # Copy original if no trimming needed
    
    if frames2 > min_frames:
        trim_video(video2_path, output2_path, min_frames)
    else:
        os.rename(video2_path, output2_path)  # Copy original if no trimming needed
    
    print("Videos processed successfully!")

if __name__ == "__main__":
    video1 = r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\1ST SURVEY\PROCESSED\Front 3 persons.mp4"
    video2 = r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\1ST SURVEY\PROCESSED\Center Walking Hand Raining2.mp4"
    output1 = r"Front.mp4"
    output2 = r"Center.mp4"
    
    main(video1, video2, output1, output2)

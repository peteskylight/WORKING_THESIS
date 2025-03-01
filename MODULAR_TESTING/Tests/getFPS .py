import cv2

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the FPS
    cap.release()
    
    return fps

# Example usage
video_path = r"C:\Users\peter\Desktop\WORKING THESIS FILES\RESOURCES\ACTUAL SURVEY\MARCH 1\TestVidIndiv.mp4"
fps = get_video_fps(video_path)
if fps:
    print(f"FPS: {fps}")

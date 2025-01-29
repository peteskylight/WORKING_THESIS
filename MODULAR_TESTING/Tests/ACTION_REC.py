import numpy as np
import tensorflow as tf
import cv2
from collections import defaultdict

# Load the action recognition model
model = tf.keras.models.load_model("your_model.h5")

# Define buffer to store sequences of keypoints per person
buffer_size = 30  # The model requires 30 frames per prediction
person_keypoints = defaultdict(lambda: [])

def preprocess_keypoints(keypoints_dict):
    """
    Convert keypoints dictionary into a properly shaped numpy array for inference.
    Missing keypoints (zeros) remain unchanged.
    """
    processed_data = {}
    for person_id, keypoints in keypoints_dict.items():
        keypoints = keypoints.flatten()  # Convert (17, 2) to (34,)
        processed_data[person_id] = keypoints
    return processed_data

def predict_actions(frames_keypoints):
    """
    Predict actions based on sequential keypoints data.
    Returns a dictionary with person ID as keys and their predicted actions as values.
    """
    predictions = {}
    for person_id, keypoints_sequence in frames_keypoints.items():
        if len(keypoints_sequence) == buffer_size:
            input_data = np.array([keypoints_sequence])  # Shape (1, 30, 34)
            prediction = model.predict(input_data)[0]  # Assuming single output per person
            predicted_action = np.argmax(prediction)  # Get the action index
            predictions[person_id] = predicted_action
    return predictions

def draw_bounding_boxes(frame, detections, actions):
    """
    Draw bounding boxes with action labels on each frame.
    """
    for person_id, bbox in detections.items():
        x, y, w, h = bbox  # Assuming bbox format is (x, y, width, height)
        action_text = f"Action: {actions.get(person_id, 'Unknown')}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, action_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def process_video(video_keypoints, video_frames, detections):
    """
    Process video frames and return a list of frames with bounding boxes and action labels.
    """
    processed_frames = []
    for frame_idx, frame_keypoints in enumerate(video_keypoints):
        processed_keypoints = preprocess_keypoints(frame_keypoints)
        
        for person_id, keypoints in processed_keypoints.items():
            person_keypoints[person_id].append(keypoints)
            
            # Keep only the latest 30 frames
            if len(person_keypoints[person_id]) > buffer_size:
                person_keypoints[person_id].pop(0)
        
        # Predict actions once we have enough frames
        action_predictions = predict_actions(person_keypoints)
        
        # Draw bounding boxes on the frame
        frame = video_frames[frame_idx]
        frame = draw_bounding_boxes(frame, detections[frame_idx], action_predictions)
        
        processed_frames.append(frame)
    
    return processed_frames

# utils/preprocessing.py

import tensorflow as tf
import numpy as np
import cv2
import tempfile

def preprocess_video(video_bytes):
    """
    Preprocesses video bytes to the format expected by the model.

    Args:
        video_bytes: Raw video data in bytes.

    Returns:
        Numpy array of preprocessed frames.
    """
    n_frames = 8
    frame_step = 10
    output_size = (224, 224)

    # Save bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp:
        temp.write(video_bytes)
        temp_path = temp.name

    # Extract frames using OpenCV
    cap = cv2.VideoCapture(temp_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > total_frames:
        start = 0
    else:
        max_start = total_frames - need_length
        start = np.random.randint(0, max_start + 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, output_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        # Move to the next frame_step
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_step - 1)
    cap.release()

    # Convert to numpy array and normalize
    frames = np.array(frames, dtype=np.float32) / 255.0
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension

    return frames

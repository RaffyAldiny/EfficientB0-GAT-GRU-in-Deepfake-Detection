# utils/preprocess.py
import os
import cv2

def preprocess_video(video_path, output_folder):
    """
    Extracts frames from a video and saves them as images.
    """
    os.makedirs(output_folder, exist_ok=True)
    if len(os.listdir(output_folder)) > 0:
        print(f"Skipping already preprocessed video: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return

    frame_count = 0
    print(f"Processing video: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            resized_frame = cv2.resize(frame, (224, 224))
            frame_path = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, resized_frame)
            frame_count += 1
        except Exception as e:
            print(f"Warning: Failed to process frame {frame_count} in {video_path}: {e}")
            continue
    cap.release()
    print(f"Finished processing video: {video_path} with {frame_count} frames.")

def preprocess_dataset(input_folder, output_folder):
    """
    Preprocess all videos in the dataset.
    """
    os.makedirs(output_folder, exist_ok=True)
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(root, file)
                video_output_folder = os.path.join(output_folder, os.path.splitext(file)[0])
                preprocess_video(video_path, video_output_folder)

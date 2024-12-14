# utils/preprocess.py

import os
import random
import cv2
from shutil import copytree, ignore_patterns
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def resize_with_padding(frame, target_size=(224, 224)):
    """
    Resize a frame while maintaining aspect ratio and pad with black borders if necessary.

    Parameters:
    - frame: The input image frame as a NumPy array.
    - target_size: Desired output size as a tuple (width, height).

    Returns:
    - new_frame: The resized and padded frame as a NumPy array.
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h))
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]  # Black padding
    new_frame = cv2.copyMakeBorder(
        resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_frame

def preprocess_video(args):
    """
    Preprocess a single video: extract frames, resize with padding, pad frames by duplicating the last frame.

    Parameters:
    - args: Tuple containing (video_path, output_folder, target_fps, max_duration, target_frames)
    """
    video_path, output_folder, target_fps, max_duration, target_frames = args
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if already preprocessed
    if len(os.listdir(output_folder)) >= target_frames:
        print(f"Skipping already preprocessed video: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        print(f"Warning: Original FPS is 0 for video: {video_path}. Skipping.")
        cap.release()
        return

    # Calculate total frames to extract based on target_fps and max_duration
    total_extracted_frames = min(int(original_fps * max_duration), target_frames)
    timestamps = [i / target_fps for i in range(target_frames)]
    saved_count = 0
    frames = []

    print(f"Processing video: {video_path}")
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            try:
                resized_frame = resize_with_padding(frame, (224, 224))
                frames.append(resized_frame)
                saved_count += 1
            except Exception as e:
                print(f"Warning: Failed to process frame at {ts} seconds in {video_path}: {e}")
        else:
            break

    # Pad frames by duplicating the last frame if necessary
    while saved_count < target_frames:
        if frames:
            frames.append(frames[-1].copy())
            saved_count += 1
        else:
            # If no frames were read, skip padding to avoid adding black frames
            print(f"Warning: No frames extracted from {video_path}. Cannot pad frames.")
            break

    # Save frames only if the desired number of frames is achieved
    if len(frames) == target_frames:
        for idx, frame in enumerate(frames):
            frame_path = os.path.join(output_folder, f"frame_{idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
        print(f"Finished processing video: {video_path} with {saved_count} frames.")
    else:
        print(f"Warning: Video {video_path} has only {len(frames)} frames after padding. Expected {target_frames} frames.")

    cap.release()

def create_balanced_subset(all_real_folders, all_fake_folders, subset_output_folder, proportion=0.5):
    """
    Create a globally balanced subset with equal number of real and fake videos.

    Parameters:
    - all_real_folders: List of directories containing preprocessed real videos.
    - all_fake_folders: List of directories containing preprocessed fake videos.
    - subset_output_folder: Directory to save the balanced subset.
    - proportion: Proportion of each class to include (e.g., 0.5 for 50%).
    """
    os.makedirs(subset_output_folder, exist_ok=True)

    # Collect all real and fake video folders
    real_videos = []
    for folder in all_real_folders:
        real_videos.extend([os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))])

    fake_videos = []
    for folder in all_fake_folders:
        fake_videos.extend([os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))])

    # Calculate the number of videos to include
    num_real_subset = int(len(real_videos) * proportion)
    num_fake_subset = int(len(fake_videos) * proportion)

    # Adjust if not enough videos
    if num_real_subset > len(real_videos):
        print("Warning: Requested more real videos than available. Selecting all real videos.")
        selected_real = real_videos
    else:
        selected_real = random.sample(real_videos, num_real_subset)

    if num_fake_subset > len(fake_videos):
        print("Warning: Requested more fake videos than available. Selecting all fake videos.")
        selected_fake = fake_videos
    else:
        selected_fake = random.sample(fake_videos, num_fake_subset)

    # Copy selected videos to the subset directory
    for video_path in selected_real + selected_fake:
        video_name = os.path.basename(video_path)
        dest = os.path.join(subset_output_folder, video_name)
        try:
            copytree(video_path, dest, dirs_exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to copy {video_path} to {subset_output_folder}: {e}")

    print(f"Created balanced subset with {len(selected_real)} real and {len(selected_fake)} fake videos in {subset_output_folder}.")

def preprocess_dataset():
    """
    Preprocess all videos in the specified directories with fixed parameters and create a globally balanced subset.
    """
    input_folders = [
        "data/Celeb-real",
        "data/Celeb-synthesis",
        "data/YouTube-real"
    ]
    output_folders = [
        "data/preprocessed/Celeb-real",
        "data/preprocessed/Celeb-synthesis",
        "data/preprocessed/YouTube-real"
    ]

    target_fps = 10
    max_duration = 10
    target_frames = 100

    # Prepare arguments for multiprocessing
    preprocess_args = []
    for input_folder, output_folder in zip(input_folders, output_folders):
        if not os.path.exists(input_folder):
            print(f"Error: Input folder {input_folder} does not exist. Skipping.")
            continue
        video_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".mp4", ".avi", ".mov"))]
        if not video_files:
            print(f"No video files found in {input_folder}. Skipping.")
            continue
        print(f"\nPreprocessing videos in {input_folder}...")
        for video in tqdm(video_files, desc=f"Processing {input_folder}", unit="video"):
            video_path = os.path.join(input_folder, video)
            video_output_folder = os.path.join(output_folder, os.path.splitext(video)[0])
            preprocess_args.append((video_path, video_output_folder, target_fps, max_duration, target_frames))

    # Use multiprocessing for faster preprocessing
    num_processes = max(1, cpu_count() - 1)
    with Pool(processes=num_processes) as pool:
        pool.map(preprocess_video, preprocess_args)

    print("\nFinished preprocessing all videos.")

    # Create globally balanced subset
    # Define real and fake preprocessed folders
    real_preprocessed_folders = [
        "data/preprocessed/Celeb-real",
        "data/preprocessed/YouTube-real"
    ]
    fake_preprocessed_folders = [
        "data/preprocessed/Celeb-synthesis"
    ]
    subset_output_folder = "data/preprocessed/balanced_subset"

    create_balanced_subset(
        all_real_folders=real_preprocessed_folders,
        all_fake_folders=fake_preprocessed_folders,
        subset_output_folder=subset_output_folder,
        proportion=0.5  # 50% of available real and fake videos
    )

def main():
    """
    Main function to initiate preprocessing.
    """
    preprocess_dataset()

if __name__ == "__main__":
    main()

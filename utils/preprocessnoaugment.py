#utils/preprocessnoaugment.py
import os
import random
import cv2
from shutil import copytree
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, current_process
import torch
import logging
import gc

from facenet_pytorch import MTCNN

# -----------------------------------------------------------
# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
# -----------------------------------------------------------

# Configure logging
logging.basicConfig(
    filename='preprocess_face.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Reduce OpenCV threading to prevent memory issues
cv2.setNumThreads(1)

# Configuration for face detection and cropping
FACE_MARGIN = 0.2  # 20% margin around detected face (adjust as needed)

########################################
# Utility Functions
########################################

def force_resize_no_padding(image, target_size=(224, 224)):
    """
    Forcefully resize the image to target_size, ignoring aspect ratio.
    No black borders are added.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def central_crop(image):
    """
    Return a centered square crop of the image.
    If image is not square, the largest possible centered square is returned.
    """
    h, w = image.shape[:2]
    if h > w:
        top = (h - w) // 2
        return image[top:top+w, :]
    else:
        left = (w - h) // 2
        return image[:, left:left+h]

def crop_face_with_margin(image, margin=FACE_MARGIN):
    """
    Detect the face in the image using MTCNN and return a crop of the image that
    covers the detected face expanded by a margin. If no face is detected, return
    a central crop.
    
    Args:
        image (np.array): The input image in BGR format.
        margin (float): Fractional margin to expand the detected face box.
        
    Returns:
        np.array: Cropped image.
    """
    # Convert to RGB for MTCNN
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, _ = face_detector.detect(rgb_image)
    
    if boxes is not None and len(boxes) > 0:
        # Use the first detected face
        box = boxes[0]  # [x1, y1, x2, y2]
        w_box = box[2] - box[0]
        h_box = box[3] - box[1]
        new_x1 = max(0, int(box[0] - margin * w_box))
        new_y1 = max(0, int(box[1] - margin * h_box))
        new_x2 = min(image.shape[1], int(box[2] + margin * w_box))
        new_y2 = min(image.shape[0], int(box[3] + margin * h_box))
        crop = image[new_y1:new_y2, new_x1:new_x2]
        return crop
    else:
        # Fallback: return a centered square crop
        return central_crop(image)

########################################
# Global Variable for Multiprocessing
########################################
face_detector = None

def initializer():
    """
    Initialize the global MTCNN face detector.
    """
    global face_detector, device
    # Use GPU if available; otherwise, use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    face_detector = MTCNN(keep_all=False, device=device)
    logging.info(f"Face detector initialized in process {current_process().name} on device {device}")

########################################
# Video Preprocessing
########################################

def preprocess_video(args):
    """
    Preprocess a single video by:
      1. Extracting frames at specified indices.
      2. For each frame, detect the face using MTCNN.
      3. Crop the image to include the face (with extra margin) or do a central crop.
      4. Force resize the resulting crop to 224x224.
      5. Save the processed frames as JPEG images.
    """
    video_path, output_folder, target_fps, max_duration, target_frames = args
    os.makedirs(output_folder, exist_ok=True)

    # Skip if video is already processed
    if len(os.listdir(output_folder)) >= target_frames:
        logging.info(f"Skipping already preprocessed video: {video_path}")
        return

    try:
        cap = cv2.VideoCapture(video_path)
    except Exception as e:
        logging.error(f"Cannot open video file: {video_path}, Error: {e}")
        return

    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        return

    try:
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            logging.warning(f"Invalid FPS ({original_fps}) for video: {video_path}. Skipping.")
            cap.release()
            return

        total_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), target_fps * max_duration))
        frame_indices = np.linspace(0, total_frames - 1, num=target_frames, dtype=int)

        processed_frames = []
        logging.info(f"Processing video: {video_path} with {len(frame_indices)} frames.")

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning(f"Frame {idx} could not be read in {video_path}. Using last valid frame.")
                if processed_frames:
                    processed_frames.append(processed_frames[-1].copy())
                else:
                    processed_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                continue

            # Crop to the face (with margin) or use central crop if no face found
            cropped = crop_face_with_margin(frame, margin=FACE_MARGIN)
            # Force resize to 224x224
            resized = force_resize_no_padding(cropped, (224, 224))
            # Convert from BGR to RGB for consistency
            final_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            processed_frames.append(final_rgb)

        cap.release()

        # If we extracted fewer frames, duplicate the last frame until we have target_frames
        while len(processed_frames) < target_frames:
            processed_frames.append(processed_frames[-1].copy() if processed_frames else np.zeros((224, 224, 3), dtype=np.uint8))

        # Save frames as JPEG images
        if len(processed_frames) == target_frames:
            for idx, frame_img in enumerate(processed_frames):
                frame_path = os.path.join(output_folder, f"frame_{idx:05d}.jpg")
                # Convert back to BGR for saving with OpenCV
                frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(frame_path, frame_bgr)
            logging.info(f"Finished processing {video_path} with {len(processed_frames)} frames.")
        else:
            logging.warning(f"{video_path} has {len(processed_frames)} frames, expected {target_frames}.")

    except Exception as e:
        logging.error(f"Unexpected error while processing {video_path}: {e}")
        cap.release()

    # Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

########################################
# Create Balanced Subset
########################################

def create_balanced_subset(all_real_folders, all_fake_folders, subset_output_folder, proportion=1.0):
    """
    Create a globally balanced subset of real and fake videos by copying folders.
    """
    os.makedirs(subset_output_folder, exist_ok=True)

    real_videos = []
    for folder in all_real_folders:
        if not os.path.exists(folder):
            logging.warning(f"Real folder not found: {folder}")
            continue
        real_videos_in_folder = [os.path.join(folder, v) for v in os.listdir(folder)
                                 if os.path.isdir(os.path.join(folder, v))]
        real_videos.extend(real_videos_in_folder)
        logging.info(f"Found {len(real_videos_in_folder)} real videos in {folder}.")

    fake_videos = []
    for folder in all_fake_folders:
        if not os.path.exists(folder):
            logging.warning(f"Fake folder not found: {folder}")
            continue
        fake_videos_in_folder = [os.path.join(folder, v) for v in os.listdir(folder)
                                 if os.path.isdir(os.path.join(folder, v))]
        fake_videos.extend(fake_videos_in_folder)
        logging.info(f"Found {len(fake_videos_in_folder)} fake videos in {folder}.")

    logging.info(f"Total real videos found: {len(real_videos)}")
    logging.info(f"Total fake videos found: {len(fake_videos)}")

    if proportion < 1.0:
        num_real_subset = int(len(real_videos) * proportion)
        num_fake_subset = int(len(fake_videos) * proportion)
        selected_real = real_videos if num_real_subset > len(real_videos) else random.sample(real_videos, num_real_subset)
        selected_fake = fake_videos if num_fake_subset > len(fake_videos) else random.sample(fake_videos, num_fake_subset)
    else:
        selected_real = real_videos
        selected_fake = fake_videos

    logging.info(f"Selected {len(selected_real)} real videos for the subset.")
    logging.info(f"Selected {len(selected_fake)} fake videos for the subset.")

    for video_path in selected_real + selected_fake:
        video_name = os.path.basename(video_path)
        dest = os.path.join(subset_output_folder, video_name)
        try:
            copytree(video_path, dest, dirs_exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to copy {video_path} to {subset_output_folder}: {e}")

    logging.info(f"Created balanced subset with {len(selected_real)} real and {len(selected_fake)} fake videos in {subset_output_folder}.")

########################################
# Main Dataset Preprocessing
########################################

def preprocess_dataset():
    """
    Main function to preprocess the dataset:
      1. Determine how many videos to process from each folder.
      2. For each selected video, extract frames and focus on the face (with margin) cropping.
      3. Resize the crops to 224x224.
      4. Create a balanced subset of preprocessed data.
    """
    # Set ratios for real/fake videos (adjust as needed)
    real_video_ratio = 1.0      # Process all real videos
    fake_video_ratio = 0.1579   # Process ~15.79% of fake videos

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

    # Frame sampling configuration
    target_fps = 8       # Frames per second to sample
    max_duration = 10    # Process up to 10 seconds per video
    target_frames = 80   # Total frames per video

    # No YOLO model is needed now; we only use the face detector.
    # Determine device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_real_videos = 0
    total_fake_videos = 0

    for input_folder in input_folders:
        if not os.path.exists(input_folder):
            logging.error(f"Input folder {input_folder} does not exist. Skipping.")
            continue
        video_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".mp4", ".avi", ".mov"))]
        if "real" in input_folder.lower():
            total_real_videos += len(video_files)
        elif "synthesis" in input_folder.lower() or "fake" in input_folder.lower():
            total_fake_videos += len(video_files)

    logging.info(f"Total real videos available: {total_real_videos}")
    logging.info(f"Total fake videos available: {total_fake_videos}")

    num_real_to_process = int(total_real_videos * real_video_ratio)
    num_fake_to_process = int(total_fake_videos * fake_video_ratio)

    real_processed_count = 0
    fake_processed_count = 0
    preprocess_args = []

    for input_folder, output_folder in zip(input_folders, output_folders):
        if not os.path.exists(input_folder):
            logging.error(f"Input folder {input_folder} does not exist. Skipping.")
            continue
        video_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".mp4", ".avi", ".mov"))]
        if not video_files:
            logging.warning(f"No video files found in {input_folder}. Skipping.")
            continue

        is_real = "real" in input_folder.lower()
        is_fake = "synthesis" in input_folder.lower() or "fake" in input_folder.lower()

        if is_real:
            real_remaining = num_real_to_process - real_processed_count
            if real_remaining <= 0:
                logging.info(f"No more real videos needed from {input_folder}")
                continue
            num_to_take = min(real_remaining, len(video_files))
            sampled_videos = random.sample(video_files, num_to_take)
            real_processed_count += len(sampled_videos)
        elif is_fake:
            fake_remaining = num_fake_to_process - fake_processed_count
            if fake_remaining <= 0:
                logging.info(f"No more fake videos needed from {input_folder}")
                continue
            num_to_take = min(fake_remaining, len(video_files))
            sampled_videos = random.sample(video_files, num_to_take)
            fake_processed_count += len(sampled_videos)
        else:
            sampled_videos = []

        if sampled_videos:
            logging.info(f"Preprocessing {len(sampled_videos)} videos in {input_folder}")

        for video in sampled_videos:
            video_path = os.path.join(input_folder, video)
            video_output_folder = os.path.join(output_folder, os.path.splitext(video)[0])
            preprocess_args.append((video_path, video_output_folder, target_fps, max_duration, target_frames))

    logging.info(f"Total real videos to be processed: {real_processed_count}")
    logging.info(f"Total fake videos to be processed: {fake_processed_count}")

    # Process videos using a single process (to reduce memory overhead)
    num_processes = 4
    with Pool(processes=num_processes, initializer=initializer) as pool:
        list(tqdm(pool.imap_unordered(preprocess_video, preprocess_args),
                  total=len(preprocess_args),
                  desc="Processing videos"))

    logging.info("Finished preprocessing selected videos.")

    real_preprocessed_folders = ["data/preprocessed/Celeb-real", "data/preprocessed/YouTube-real"]
    fake_preprocessed_folders = ["data/preprocessed/Celeb-synthesis"]
    subset_output_folder = "data/preprocessed/balanced_subset"

    create_balanced_subset(
        all_real_folders=real_preprocessed_folders,
        all_fake_folders=fake_preprocessed_folders,
        subset_output_folder=subset_output_folder,
        proportion=1.0
    )

def main():
    preprocess_dataset()

if __name__ == "__main__":
    main()

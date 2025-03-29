import os
import random
import cv2
from shutil import copytree
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, current_process
import torch
import logging
import gc  # Garbage collector

# YOLO (Ultralytics) and MTCNN
from ultralytics import YOLO
from facenet_pytorch import MTCNN

# -----------------------------------------------------------
# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
# -----------------------------------------------------------

# Configure logging
logging.basicConfig(
    filename='preprocess.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Reduce OpenCV threading to prevent memory issues
cv2.setNumThreads(1)

########################################
# Configuration for Detection
########################################

# Maximum dimension for YOLO inference to speed up processing
MAX_YOLO_SIZE = 640  # You can adjust this value for a trade-off between speed and accuracy

########################################
# Utility Functions
########################################

def load_yolo_model(model_path, device):
    """
    Load the YOLO model for person detection.
    Optionally convert to half precision for faster inference on CUDA.
    """
    model = YOLO(model_path)
    model.to(device)
    if device == "cuda":
        model.half()  # use FP16 for speed and lower memory usage
    return model

def unify_boxes(box1, box2):
    """
    Given two bounding boxes in [x1, y1, x2, y2] format,
    return the bounding box that covers both (their union).
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return [x1, y1, x2, y2]

def force_resize_no_padding(image, target_size=(224, 224)):
    """
    Forcefully resize the image to target_size, ignoring aspect ratio.
    No black borders are added.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

########################################
# Global Variables for Multiprocessing
########################################
yolo_model = None
face_detector = None

def initializer(model_path, device):
    """
    Initialize global models for each process.
    """
    global yolo_model, face_detector
    yolo_model = load_yolo_model(model_path, device)
    face_detector = MTCNN(device=device, keep_all=False)  # detect only the first face
    logging.info(f"Models loaded in process {current_process().name} on device {device}")

########################################
# Video Preprocessing
########################################

def preprocess_video(args):
    """
    Preprocess a single video by:
      1. Extracting frames at specified indices.
      2. Downscaling the frame for YOLO detection.
      3. Person detection with YOLO (using the downscaled image).
      4. Converting bounding boxes back to the original resolution.
      5. Face detection with MTCNN to ensure the face is included.
      6. Crop the union bounding box (person + face).
      7. Force resize to 224x224 (no padding).
      8. Save frames as JPEG images.
    """
    (video_path, output_folder, target_fps, max_duration, target_frames) = args
    os.makedirs(output_folder, exist_ok=True)

    # Skip if already processed
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

        frames = []
        logging.info(f"Processing video: {video_path} with {len(frame_indices)} frames.")

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning(f"Frame {idx} could not be read in {video_path}. Using last valid frame.")
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                continue

            original_h, original_w = frame.shape[:2]

            # Downscale for YOLO if frame is larger than MAX_YOLO_SIZE
            scale_factor = 1.0
            if max(original_h, original_w) > MAX_YOLO_SIZE:
                scale_factor = MAX_YOLO_SIZE / float(max(original_h, original_w))
                d_w = int(original_w * scale_factor)
                d_h = int(original_h * scale_factor)
                yolo_frame = cv2.resize(frame, (d_w, d_h), interpolation=cv2.INTER_AREA)
            else:
                yolo_frame = frame.copy()

            # 1) Person detection with YOLO (using the downscaled frame)
            with torch.no_grad():
                results = yolo_model(yolo_frame, verbose=False)
            if not results or len(results) == 0:
                final_crop = force_resize_no_padding(frame)
                frames.append(cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB))
                continue

            result = results[0]
            if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                final_crop = force_resize_no_padding(frame)
                frames.append(cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB))
                continue

            # 2) Filter for person class in the downscaled coordinates
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            detections = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] in downscaled space

            person_class_id = 0
            if hasattr(yolo_model.model, 'names'):
                for cid, cname in yolo_model.model.names.items():
                    if cname == 'person':
                        person_class_id = cid
                        break

            mask = (confs >= 0.5) & (classes == person_class_id)
            person_boxes = detections[mask]

            if len(person_boxes) == 0:
                final_crop = force_resize_no_padding(frame)
                frames.append(cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB))
                continue

            # Choose the largest bounding box by area (in downscaled coordinates)
            areas = (person_boxes[:, 2] - person_boxes[:, 0]) * (person_boxes[:, 3] - person_boxes[:, 1])
            largest_idx = np.argmax(areas)
            bx1, by1, bx2, by2 = person_boxes[largest_idx].astype(int)
            # Map coordinates back to original resolution
            x1 = int(bx1 / scale_factor)
            y1 = int(by1 / scale_factor)
            x2 = int(bx2 / scale_factor)
            y2 = int(by2 / scale_factor)

            # Clamp to valid image boundaries
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            person_crop = frame[y1:y2, x1:x2]

            # 3) Face detection inside person crop
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            face_boxes, _ = face_detector.detect(person_rgb)
            if face_boxes is not None and len(face_boxes) > 0:
                fx1, fy1, fx2, fy2 = face_boxes[0].astype(int)
                # Convert face coordinates to original frame space
                fx1_global = x1 + fx1
                fy1_global = y1 + fy1
                fx2_global = x1 + fx2
                fy2_global = y1 + fy2
                union_box = unify_boxes([x1, y1, x2, y2],
                                        [fx1_global, fy1_global, fx2_global, fy2_global])
                x1u, y1u, x2u, y2u = [int(v) for v in union_box]
                # Clamp again
                x1u = max(0, min(x1u, w - 1))
                y1u = max(0, min(y1u, h - 1))
                x2u = max(0, min(x2u, w - 1))
                y2u = max(0, min(y2u, h - 1))
                final_crop = frame[y1u:y2u, x1u:x2u]
            else:
                final_crop = person_crop

            # 4) Force resize to exactly 224x224
            final_crop = force_resize_no_padding(final_crop, (224, 224))
            final_rgb = cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB)
            frames.append(final_rgb)

        cap.release()

        # If fewer frames were extracted, pad by duplicating the last frame
        while len(frames) < target_frames:
            frames.append(frames[-1].copy() if frames else np.zeros((224, 224, 3), dtype=np.uint8))

        # Save frames as JPEGs
        if len(frames) == target_frames:
            for idx, frame_img in enumerate(frames):
                frame_path = os.path.join(output_folder, f"frame_{idx:05d}.jpg")
                frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(frame_path, frame_bgr)
            logging.info(f"Finished processing {video_path} with {len(frames)} frames.")
        else:
            logging.warning(f"{video_path} has {len(frames)} frames, expected {target_frames}.")

    except MemoryError as me:
        logging.error(f"MemoryError while processing {video_path}: {me}")
        cap.release()
    except cv2.error as ce:
        logging.error(f"cv2 error while processing {video_path}: {ce}")
        cap.release()
    except Exception as e:
        logging.error(f"Unexpected error while processing {video_path}: {e}")
        cap.release()

    # Clean up: run garbage collection and clear CUDA cache if applicable
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
      2. For each selected video, extract frames, detect person+face, crop+resize to 224x224.
      3. Create a balanced subset of preprocessed data.
    """
    real_video_ratio = 1.0      # Process 100% of real videos
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

    target_fps = 8       # Frames per second to sample
    max_duration = 10    # Process up to 10 seconds per video
    target_frames = 80   # Total frames per video

    yolo_model_path = "yolov5su.pt"  # Update with your YOLO model file
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

    # Using a single process to reduce memory bloat
    num_processes = 1
    with Pool(processes=num_processes, initializer=initializer, initargs=(yolo_model_path, device)) as pool:
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

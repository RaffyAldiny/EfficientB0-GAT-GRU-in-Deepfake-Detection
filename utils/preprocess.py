import os
import random
import cv2
from shutil import copytree
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, current_process
import albumentations as A
from ultralytics import YOLO
import torch
import logging
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

# Reduce OpenCV threading to prevent memory issues and improve speed consistency
cv2.setNumThreads(1)

def load_yolo_model(model_path, device):
    """
    Load the YOLO model for person detection.
    """
    model = YOLO(model_path)
    model.to(device)
    return model

# Global variables for multiprocessing
yolo_model = None
face_detector = None

def initializer(model_path, device):
    """
    Initialize global models for each process.
    """
    global yolo_model, face_detector
    yolo_model = load_yolo_model(model_path, device)
    face_detector = MTCNN(device=device, keep_all=False)
    logging.info(f"Models loaded in process {current_process().name} on device {device}")

def resize_with_padding(frame, target_size=(224, 224)):
    """
    Resize an image while maintaining aspect ratio and pad with black borders.
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_frame = cv2.copyMakeBorder(
        resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_frame

def get_augmentation_pipelines():
    """
    Define augmentation pipelines using ReplayCompose.
    Includes a "no augmentation" pipeline for retaining original frames.
    """
    transforms = [
        A.ReplayCompose([]),  # No augmentation (retain original frame)
        A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        ]),
        A.ReplayCompose([
            A.VerticalFlip(p=0.3),
            A.RandomGamma(p=0.5),
            A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.2),
            A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.3),
            A.Sharpen(alpha=(0.0, 0.3), lightness=(0.75, 1.5), p=0.3),
        ]),
        A.ReplayCompose([
            A.RandomSizedCrop(min_max_height=(200, 220), height=224, width=224, p=0.5),
            A.Blur(blur_limit=3, p=0.2),
            A.HueSaturationValue(p=0.25),
            A.ToGray(p=0.2),
            A.RandomScale(scale_limit=0.1, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ]),
    ]
    return transforms

def preprocess_video(args):
    """
    Preprocess a single video:
    1. Extract frames
    2. Detect person and face
    3. Apply augmentations
    4. Save frames
    """
    video_path, output_folder, target_fps, max_duration, target_frames = args
    os.makedirs(output_folder, exist_ok=True)

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

        augmentation_pipelines = get_augmentation_pipelines()
        selected_transform = random.choice(augmentation_pipelines)
        try:
            # Apply augmentation once to get replay params
            sample_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            augmented = selected_transform(image=sample_frame)
            replay_params = augmented['replay']
        except Exception as e:
            logging.error(f"Failed to initialize augmentation for {video_path}: {e}")
            cap.release()
            return

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

            # Resizing down can speed YOLO but may reduce detection accuracy
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

            try:
                # YOLO detection
                results = yolo_model(frame, verbose=False)
                if not results or len(results) == 0:
                    frames.append(frames[-1].copy() if frames else resize_with_padding(frame))
                    continue

                result = results[0]
                if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                    frames.append(frames[-1].copy() if frames else resize_with_padding(frame))
                    continue

                # Filter for person class
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                detections = result.boxes.xyxy.cpu().numpy()

                person_class_id = 0
                if hasattr(yolo_model.model, 'names'):
                    names = yolo_model.model.names
                    for cid, cname in names.items():
                        if cname == 'person':
                            person_class_id = cid
                            break

                mask = (confs >= 0.5) & (classes == person_class_id)
                detections = detections[mask]

                if len(detections) == 0:
                    frames.append(frames[-1].copy() if frames else resize_with_padding(frame))
                    continue

                # Choose largest bounding box
                areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
                largest_idx = np.argmax(areas)
                x1, y1, x2, y2 = detections[largest_idx].astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    frames.append(frames[-1].copy() if frames else resize_with_padding(frame))
                    continue

                # Face detection
                person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                box, _ = face_detector.detect(person_rgb)
                if box is not None and len(box) > 0:
                    fx1, fy1, fx2, fy2 = box[0].astype(int)
                    fx1 = max(0, fx1)
                    fy1 = max(0, fy1)
                    fx2 = min(person_crop.shape[1], fx2)
                    fy2 = min(person_crop.shape[0], fy2)
                    face_crop = person_rgb[fy1:fy2, fx1:fx2]
                    final_crop = face_crop if face_crop.size > 0 else person_rgb
                else:
                    final_crop = person_rgb

                # Resize and pad final image
                resized_frame = resize_with_padding(final_crop, (224, 224))

                # Apply stored augmentation params
                augmented_frame = A.ReplayCompose.replay(replay_params, image=resized_frame)['image']

                frames.append(augmented_frame)

            except Exception as e:
                logging.error(f"Error processing frame {idx} in {video_path}: {e}")
                frames.append(frames[-1].copy() if frames else resize_with_padding(frame))

        cap.release()

        # Pad frames if needed
        while len(frames) < target_frames:
            frames.append(frames[-1].copy() if frames else np.zeros((224, 224, 3), dtype=np.uint8))

        # Save frames
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

def create_balanced_subset(all_real_folders, all_fake_folders, subset_output_folder, proportion=1.0):
    """
    Create a globally balanced subset of real and fake videos.
    """
    os.makedirs(subset_output_folder, exist_ok=True)

    real_videos = []
    for folder in all_real_folders:
        if not os.path.exists(folder):
            logging.warning(f"Real folder not found: {folder}")
            continue
        real_videos_in_folder = [
            os.path.join(folder, v) for v in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, v))
        ]
        real_videos.extend(real_videos_in_folder)
        logging.info(f"Found {len(real_videos_in_folder)} real videos in {folder}.")

    fake_videos = []
    for folder in all_fake_folders:
        if not os.path.exists(folder):
            logging.warning(f"Fake folder not found: {folder}")
            continue
        fake_videos_in_folder = [
            os.path.join(folder, v) for v in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, v))
        ]
        fake_videos.extend(fake_videos_in_folder)
        logging.info(f"Found {len(fake_videos_in_folder)} fake videos in {folder}.")

    logging.info(f"Total real videos found: {len(real_videos)}")
    logging.info(f"Total fake videos found: {len(fake_videos)}")

    if proportion < 1.0:
        num_real_subset = int(len(real_videos) * proportion)
        num_fake_subset = int(len(fake_videos) * proportion)

        if num_real_subset > len(real_videos):
            logging.warning("Requested more real videos than available, using all real.")
            selected_real = real_videos
        else:
            selected_real = random.sample(real_videos, num_real_subset)

        if num_fake_subset > len(fake_videos):
            logging.warning("Requested more fake videos than available, using all fake.")
            selected_fake = fake_videos
        else:
            selected_fake = random.sample(fake_videos, num_fake_subset)
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

def preprocess_dataset():
    """
    Main function to preprocess the dataset.
    """

    # Set sampling ratios for real and fake videos here:
    real_video_ratio = 0.80  # For example, 25% of all real videos
    fake_video_ratio = 0.20  # For example, 25% of all fake videos

    input_folders = [
        "data/Celeb-real",
        "data/Celeb-synthesis",
        "data/YouTube-real"
    ]
    output_folders = [
        "data/temp/Celeb-real",
        "data/temp/Celeb-synthesis",
        "data/temp/YouTube-real"
    ]

    target_fps = 8
    max_duration = 10
    target_frames = 80
    # Update your YOLO model path here if needed
    yolo_model_path = "yolov5su.pt"

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocess_args = []
    total_real_videos = 0
    total_fake_videos = 0
    processed_real_videos = 0
    processed_fake_videos = 0

    # First, gather total counts of real and fake videos
    for input_folder in input_folders:
        if not os.path.exists(input_folder):
            logging.error(f"Input folder {input_folder} does not exist. Skipping.")
            continue
        video_files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ]
        total_videos = len(video_files)

        if "real" in input_folder.lower():
            total_real_videos += total_videos
        elif "synthesis" in input_folder.lower() or "fake" in input_folder.lower():
            total_fake_videos += total_videos

    logging.info(f"Total real videos available: {total_real_videos}")
    logging.info(f"Total fake videos available: {total_fake_videos}")

    # Determine the number of real and fake videos to process based on the ratios
    num_real_to_process = int(total_real_videos * real_video_ratio)
    num_fake_to_process = int(total_fake_videos * fake_video_ratio)

    # Keep track of how many we've processed so far
    real_processed_count = 0
    fake_processed_count = 0

    # Now process the videos from each folder
    for input_folder, output_folder in zip(input_folders, output_folders):
        if not os.path.exists(input_folder):
            logging.error(f"Input folder {input_folder} does not exist. Skipping.")
            continue

        video_files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ]

        if not video_files:
            logging.warning(f"No video files found in {input_folder}. Skipping.")
            continue

        is_real = "real" in input_folder.lower()
        is_fake = "synthesis" in input_folder.lower() or "fake" in input_folder.lower()

        if is_real:
            # Calculate how many real videos we can still process
            real_remaining = num_real_to_process - real_processed_count
            if real_remaining <= 0:
                logging.info(f"No more real videos needed from {input_folder}")
                continue
            # If real_remaining > available videos, just take them all
            num_to_take = min(real_remaining, len(video_files))
            sampled_videos = random.sample(video_files, num_to_take)
            real_processed_count += len(sampled_videos)
        elif is_fake:
            # Calculate how many fake videos we can still process
            fake_remaining = num_fake_to_process - fake_processed_count
            if fake_remaining <= 0:
                logging.info(f"No more fake videos needed from {input_folder}")
                continue
            num_to_take = min(fake_remaining, len(video_files))
            sampled_videos = random.sample(video_files, num_to_take)
            fake_processed_count += len(sampled_videos)
        else:
            # If folder is neither real nor fake by naming convention, skip
            sampled_videos = []

        # Update processed counts for logging
        if is_real:
            processed_real_videos += len(sampled_videos)
        elif is_fake:
            processed_fake_videos += len(sampled_videos)

        if sampled_videos:
            logging.info(f"Preprocessing {len(sampled_videos)} videos in {input_folder}.")

        for video in sampled_videos:
            video_path = os.path.join(input_folder, video)
            video_output_folder = os.path.join(output_folder, os.path.splitext(video)[0])
            preprocess_args.append((video_path, video_output_folder, target_fps, max_duration, target_frames))

    logging.info(f"Total real videos to be processed: {processed_real_videos}")
    logging.info(f"Total fake videos to be processed: {processed_fake_videos}")

    num_processes = 8
    with Pool(processes=num_processes, initializer=initializer, initargs=(yolo_model_path, device)) as pool:
        list(tqdm(pool.imap_unordered(preprocess_video, preprocess_args), total=len(preprocess_args), desc="Processing videos"))

    logging.info("Finished preprocessing selected videos.")

    # Create a balanced subset from the preprocessed data
    real_preprocessed_folders = [
        "data/temp/Celeb-real",
        "data/temp/YouTube-real"
    ]
    fake_preprocessed_folders = [
        "data/temp/Celeb-synthesis"
    ]
    subset_output_folder = "data/preprocessed/balanced_subset"

    create_balanced_subset(
        all_real_folders=real_preprocessed_folders,
        all_fake_folders=fake_preprocessed_folders,
        subset_output_folder=subset_output_folder,
        proportion=1.0  # Include all sampled data in the balanced subset
    )

def main():
    preprocess_dataset()

if __name__ == "__main__":
    main()

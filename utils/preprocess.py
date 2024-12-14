import os
import random
import cv2
from shutil import copytree
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
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
    global yolo_model, face_detector
    yolo_model = load_yolo_model(model_path, device)
    face_detector = MTCNN(device=device, keep_all=False)
    logging.info(f"Models loaded in process {os.getpid()} on device {device}")

def resize_with_padding(frame, target_size=(224, 224)):
    """
    Resize an image while maintaining aspect ratio and pad with black borders.
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
    color = [0, 0, 0]
    new_frame = cv2.copyMakeBorder(
        resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_frame

def get_augmentation_pipelines():
    """
    Use ReplayCompose to ensure consistent augmentation across all frames of a video.
    """
    # Optional: Consider adding a normalization transform here if you want:
    # A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), max_pixel_value=255.0)
    # However, it's typically better done during training rather than preprocessing.
    transforms = [
        A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
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
            A.HueSaturationValue(p=0.5),
            A.ToGray(p=0.2),
            A.RandomScale(scale_limit=0.1, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ]),
    ]
    return transforms

def preprocess_video(args):
    video_path, output_folder, target_fps, max_duration, target_frames = args
    os.makedirs(output_folder, exist_ok=True)
    
    if len(os.listdir(output_folder)) >= target_frames:
        logging.info(f"Skipping already preprocessed video: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        logging.warning(f"Original FPS is 0 for video: {video_path}. Skipping.")
        cap.release()
        return

    timestamps = [i / target_fps for i in range(target_frames)]
    saved_count = 0
    frames = []

    augmentation_pipelines = get_augmentation_pipelines()
    selected_transform = random.choice(augmentation_pipelines)
    logging.info(f"Selected augmentation pipeline for {video_path}: {selected_transform}")
    
    replay_params = None

    logging.info(f"Processing video: {video_path}")
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            try:
                # YOLO to detect person
                results = yolo_model(frame)
                if not results or len(results) == 0:
                    logging.warning(f"No YOLO results at {ts:.2f}s in {video_path}")
                    if frames:
                        frames.append(frames[-1].copy())
                        saved_count += 1
                    continue

                result = results[0]
                if not result.boxes or len(result.boxes) == 0:
                    logging.warning(f"No person detected at {ts:.2f}s in {video_path}")
                    if frames:
                        frames.append(frames[-1].copy())
                        saved_count += 1
                    continue

                # Filter detections for person class with confidence >= 0.5
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                detections = result.boxes.xyxy.cpu().numpy()

                if hasattr(yolo_model.model, 'names'):
                    names = yolo_model.model.names
                    person_class_id = None
                    for cid, cname in names.items():
                        if cname == 'person':
                            person_class_id = cid
                            break
                    if person_class_id is None:
                        logging.warning("No 'person' class found in YOLO names. Using class 0.")
                        person_class_id = 0
                else:
                    logging.warning("YOLO model has no names attribute. Using class 0 for person.")
                    person_class_id = 0

                mask = (confs >= 0.5) & (classes == person_class_id)
                detections = detections[mask]

                if len(detections) == 0:
                    logging.warning(f"No confident person at {ts:.2f}s in {video_path}")
                    if frames:
                        frames.append(frames[-1].copy())
                        saved_count += 1
                    continue

                # Choose the largest bounding box if multiple
                areas = [(d[2]-d[0])*(d[3]-d[1]) for d in detections]
                largest_idx = np.argmax(areas)
                x1, y1, x2, y2 = detections[largest_idx].astype(int)

                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    logging.warning(f"Zero-size person at {ts:.2f}s in {video_path}")
                    if frames:
                        frames.append(frames[-1].copy())
                        saved_count += 1
                    continue

                # Detect face inside person region
                person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                box, _ = face_detector.detect(person_rgb)
                if box is not None:
                    fx1, fy1, fx2, fy2 = box[0].astype(int)
                    fx1 = max(0, fx1)
                    fy1 = max(0, fy1)
                    fx2 = min(person_crop.shape[1], fx2)
                    fy2 = min(person_crop.shape[0], fy2)
                    face_crop = person_rgb[fy1:fy2, fx1:fx2]
                    if face_crop.size > 0:
                        final_crop = face_crop
                    else:
                        final_crop = person_rgb
                else:
                    final_crop = person_rgb

                # Resize and pad
                resized_frame = resize_with_padding(final_crop, (224, 224))

                # Apply augmentation consistently
                if replay_params is None:
                    augmented = selected_transform(image=resized_frame)
                    augmented_frame = augmented['image']
                    replay_params = augmented['replay']
                else:
                    augmented_frame = A.ReplayCompose.replay(replay_params, image=resized_frame)['image']

                frames.append(augmented_frame)
                saved_count += 1

            except Exception as e:
                logging.error(f"Error at {ts:.2f}s in {video_path}: {e}")
                if frames:
                    frames.append(frames[-1].copy())
                    saved_count += 1
        else:
            logging.warning(f"Unable to read frame at {ts:.2f}s in {video_path}")
            if frames:
                frames.append(frames[-1].copy())
                saved_count += 1

    cap.release()

    # Pad frames if needed
    while saved_count < target_frames:
        if frames:
            frames.append(frames[-1].copy())
            saved_count += 1
        else:
            logging.warning(f"No frames extracted from {video_path}. Cannot pad.")
            break

    # Save frames
    if len(frames) == target_frames:
        for idx, frame_img in enumerate(frames):
            frame_path = os.path.join(output_folder, f"frame_{idx:05d}.jpg")
            frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)
        logging.info(f"Finished processing {video_path} with {saved_count} frames.")
    else:
        logging.warning(f"{video_path} has {len(frames)} frames, expected {target_frames}.")

def create_balanced_subset(all_real_folders, all_fake_folders, subset_output_folder, proportion=0.5):
    """
    Create a globally balanced subset of real and fake videos.
    """
    os.makedirs(subset_output_folder, exist_ok=True)

    real_videos = []
    for folder in all_real_folders:
        if not os.path.exists(folder):
            logging.warning(f"Real folder not found: {folder}")
            continue
        real_videos.extend([os.path.join(folder, v) for v in os.listdir(folder) if os.path.isdir(os.path.join(folder, v))])

    fake_videos = []
    for folder in all_fake_folders:
        if not os.path.exists(folder):
            logging.warning(f"Fake folder not found: {folder}")
            continue
        fake_videos.extend([os.path.join(folder, v) for v in os.listdir(folder) if os.path.isdir(os.path.join(folder, v))])

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

    for video_path in selected_real + selected_fake:
        video_name = os.path.basename(video_path)
        dest = os.path.join(subset_output_folder, video_name)
        try:
            copytree(video_path, dest, dirs_exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to copy {video_path} to {subset_output_folder}: {e}")

    logging.info(f"Created balanced subset with {len(selected_real)} real and {len(selected_fake)} fake videos in {subset_output_folder}.")

def preprocess_dataset():
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

    target_fps = 10
    max_duration = 10
    target_frames = 100
    yolo_model_path = "yolov5s.pt" # Ensure you have this model accessible

    device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocess_args = []
    for input_folder, output_folder in zip(input_folders, output_folders):
        if not os.path.exists(input_folder):
            logging.error(f"Input folder {input_folder} does not exist. Skipping.")
            continue
        video_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".mp4", ".avi", ".mov"))]
        if not video_files:
            logging.warning(f"No video files found in {input_folder}. Skipping.")
            continue
        logging.info(f"\nPreprocessing videos in {input_folder}...")
        for video in tqdm(video_files, desc=f"Processing {input_folder}", unit="video"):
            video_path = os.path.join(input_folder, video)
            video_output_folder = os.path.join(output_folder, os.path.splitext(video)[0])
            preprocess_args.append((video_path, video_output_folder, target_fps, max_duration, target_frames))

    num_processes = min(4, max(1, cpu_count() - 1))
    with Pool(processes=num_processes, initializer=initializer, initargs=(yolo_model_path, device)) as pool:
        pool.map(preprocess_video, preprocess_args)

    logging.info("\nFinished preprocessing all videos.")

    # Create a balanced subset
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
        proportion=0.5
    )

def main():
    preprocess_dataset()

if __name__ == "__main__":
    main()

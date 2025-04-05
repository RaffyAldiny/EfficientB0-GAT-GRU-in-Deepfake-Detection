import os
import cv2
import torch
import mediapipe as mp
from tqdm import tqdm
from facenet_pytorch import MTCNN

# 68‑point mapping from MediaPipe’s 468 landmarks
MP_TO_68_INDICES = [
    # Jawline
    127, 34, 139, 35, 36, 37, 38, 39, 40, 41, 142, 143, 144, 145, 146, 47, 46,
    # Right brow
    107, 66, 105, 63, 70,
    # Left brow
    336, 296, 334, 293, 300,
    # Nose bridge
    168, 6, 197, 195,
    # Lower nose
    5, 4, 1, 19, 94,
    # Right eye
    33, 7, 163, 144, 145, 153,
    # Left eye
    263, 249, 390, 373, 374, 380,
    # Outer lip
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    # Inner lip
    78, 95, 88, 178, 87, 14, 317, 402
]

# 1) MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# 2) MTCNN fallback
mtcnn = MTCNN(device='cpu')

def compute_landmarks_with_mediapipe(frame):
    """Try MediaPipe on the full frame."""
    res = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm468 = res.multi_face_landmarks[0].landmark
    coords = [(lm.x, lm.y) for lm in lm468]
    coords68 = [coords[i] for i in MP_TO_68_INDICES]
    return torch.tensor(coords68, dtype=torch.float32)

def compute_landmarks(frame):
    """
    1) Run MediaPipe on full frame.
    2) If no face, detect box via MTCNN, crop, re-run MediaPipe.
    """
    # --- Try MediaPipe on full image ---
    landmarks = compute_landmarks_with_mediapipe(frame)
    if landmarks is not None:
        return landmarks

    # --- Fallback to MTCNN box + crop ---
    boxes, _ = mtcnn.detect(frame)
    if boxes is None or len(boxes) == 0:
        return None

    x1, y1, x2, y2 = boxes[0].astype(int)
    # Check for invalid box
    if x1 >= x2 or y1 >= y2:
        return None

    # Crop and re‑run MediaPipe
    face_crop = frame[y1:y2, x1:x2]
    landmarks_crop = compute_landmarks_with_mediapipe(face_crop)
    if landmarks_crop is None:
        return None

    # Map normalized coords back to original pixel space
    h_crop, w_crop = face_crop.shape[:2]
    landmarks_orig = landmarks_crop.clone()
    landmarks_orig[:, 0] = landmarks_crop[:, 0] * w_crop + x1
    landmarks_orig[:, 1] = landmarks_crop[:, 1] * h_crop + y1
    return landmarks_orig

def process_folder(folder):
    """
    Given a folder, find all .jpg/.png, run compute_landmarks(),
    collect and save as landmarks.pt
    """
    frames = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.png'))
    )
    if not frames:
        # No images here
        return

    all_lm = []
    for f in tqdm(frames, desc=os.path.basename(folder)):
        img_path = os.path.join(folder, f)
        img = cv2.imread(img_path)
        if img is None:
            continue
        lm = compute_landmarks(img)
        if lm is None:
            print(f"Landmark failure in {img_path}, using zeros")
            lm = torch.zeros((68, 2), dtype=torch.float32)
        all_lm.append(lm)

    if not all_lm:
        print(f"Skipping empty folder: {folder}")
        return

    if len(all_lm) < 10:
        print(f"WARNING: {folder} has only {len(all_lm)} frames")

    # Save stacked tensor of shape [N_frames, 68, 2]
    torch.save(torch.stack(all_lm), os.path.join(folder, "landmarks.pt"))

def main():
    root = "data/preprocessed/"

    # Walk through all nested subdirectories
    for foldername, subfolders, filenames in os.walk(root):
        # Only process if there are image files here
        if any(f.lower().endswith(('.jpg', '.png')) for f in filenames):
            process_folder(foldername)

if __name__ == "__main__":
    main()

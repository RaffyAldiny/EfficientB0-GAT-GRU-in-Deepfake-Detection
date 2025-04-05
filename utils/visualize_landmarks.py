#!/usr/bin/env python3
import os
import sys
import argparse

import cv2
import torch
import mediapipe as mp
from facenet_pytorch import MTCNN

# 1) 68‑point mapping from MediaPipe’s 468 landmarks
MP_TO_68_INDICES = [
    # Jawline
    127,34,139,35,36,37,38,39,40,41,142,143,144,145,146,47,46,
    # Right brow
    107,66,105,63,70,
    # Left brow
    336,296,334,293,300,
    # Nose bridge
    168,6,197,195,
    # Lower nose
    5,4,1,19,94,
    # Right eye
    33,7,163,144,145,153,
    # Left eye
    263,249,390,373,374,380,
    # Outer lip
    61,146,91,181,84,17,314,405,321,375,291,308,
    # Inner lip
    78,95,88,178,87,14,317,402
]

# (Optional) If you want to draw edges too, uncomment and use EDGES_68:
# EDGES_68 = [
#     *[(i, i+1) for i in range(0,16)],           # Jawline
#     *[(i, i+1) for i in range(17,21)],          # Right eyebrow
#     *[(i, i+1) for i in range(22,26)],          # Left eyebrow
#     *[(i, i+1) for i in range(27,30)],          # Nose bridge
#     *[(i, i+1) for i in range(31,35)],          # Lower nose
#     *[(i, i+1) for i in range(36,41)] + [(41,36)],  # Right eye
#     *[(i, i+1) for i in range(42,47)] + [(47,42)],  # Left eye
#     *[(i, i+1) for i in range(48,59)] + [(59,48)],  # Outer lip
#     *[(i, i+1) for i in range(60,67)] + [(67,60)]   # Inner lip
# ]

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Initialize MTCNN fallback detector
mtcnn = MTCNN(device='cpu')

def compute_landmarks_with_mediapipe(frame):
    """Run MediaPipe on full frame; returns normalized [68,2] or None."""
    res = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm468 = res.multi_face_landmarks[0].landmark
    coords = [(lm.x, lm.y) for lm in lm468]
    coords68 = [coords[i] for i in MP_TO_68_INDICES]
    return torch.tensor(coords68, dtype=torch.float32)

def compute_landmarks(frame):
    """
    1) Try MediaPipe on the full frame.
    2) If that fails, detect with MTCNN, crop, re‑run MediaPipe.
    Returns pixel coords [68,2] or None.
    """
    # Full-frame MediaPipe
    lm = compute_landmarks_with_mediapipe(frame)
    if lm is not None:
        h, w = frame.shape[:2]
        lm[:,0] *= w
        lm[:,1] *= h
        return lm

    # Fallback: MTCNN box → crop
    boxes, _ = mtcnn.detect(frame)
    if boxes is None or len(boxes)==0:
        return None
    x1,y1,x2,y2 = boxes[0].astype(int)
    if x2<=x1 or y2<=y1:
        return None

    crop = frame[y1:y2, x1:x2]
    lm_crop = compute_landmarks_with_mediapipe(crop)
    if lm_crop is None:
        return None

    h_c, w_c = crop.shape[:2]
    lm_crop[:,0] = lm_crop[:,0]*w_c + x1
    lm_crop[:,1] = lm_crop[:,1]*h_c + y1
    return lm_crop

def visualize_folder(input_folder: str, output_folder: str):
    """Overlay landmarks on every .jpg/.png in input_folder."""
    os.makedirs(output_folder, exist_ok=True)

    for fn in sorted(os.listdir(input_folder)):
        if not fn.lower().endswith(('.jpg','.png')):
            continue

        img_path = os.path.join(input_folder, fn)
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Could not read: {img_path}")
            continue

        lm = compute_landmarks(img)
        if lm is None:
            print(f"⚠️  No landmarks in {img_path}")
            continue

        pts = lm.cpu().numpy().astype(int)
        # Draw edges if desired:
        # for i,j in EDGES_68:
        #     cv2.line(img, tuple(pts[i]), tuple(pts[j]), (255,200,0), 1)

        # Draw nodes:
        for x,y in pts:
            cv2.circle(img, (x,y), radius=2, color=(0,255,0), thickness=-1)

        out_path = os.path.join(output_folder, fn)
        cv2.imwrite(out_path, img)
        print(f"✅ {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Compute & overlay 68 landmarks on all images in a folder."
    )
    parser.add_argument("input_folder",
                        help="Folder of .jpg/.png frames (no landmarks.pt needed)")
    parser.add_argument("output_folder",
                        help="Where to save overlaid images")
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"❌ Input folder not found: {args.input_folder}")
        sys.exit(1)

    visualize_folder(args.input_folder, args.output_folder)

if __name__=="__main__":
    main()

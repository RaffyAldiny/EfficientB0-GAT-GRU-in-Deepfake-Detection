import albumentations as A
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import random

def get_augmentation_pipelines():
    """
    Define augmentation pipelines using ReplayCompose.
    
    Augmentation Variations:
    1. **Noise & Contrast Enhancement** - Adds noise, adjusts brightness, contrast, and applies motion blur.
    2. **Color & Sharpness Adjustment** - Modifies hue, saturation, gamma, and sharpens the image.
    3. **Blur & Grayscale Transformation** - Applies Gaussian blur, converts to grayscale, and slightly scales the image.
    
    Returns:
    - List of (name, augmentation_pipeline) tuples.
    """
    transforms = [
        ("Noise & Contrast Enhancement", A.ReplayCompose([
            A.RandomBrightnessContrast(p=0.75),
            A.GaussNoise(std_range=(0.03, 0.05), p=1.0),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=1.0),
        ])),
        ("Color & Sharpness Adjustment", A.ReplayCompose([
            A.RandomGamma(p=0.5),
            A.HueSaturationValue(p=1.0),
            A.MedianBlur(blur_limit=3, p=0.2),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.75, 1.5), p=1.0),
        ])),
        ("Blur & Grayscale Transformation", A.ReplayCompose([
            A.GaussianBlur(blur_limit=(8, 9), p=0.5),
            A.ToGray(p=1.0),
            A.RandomScale(scale_limit=0.1, p=0.3),
        ])),
    ]
    return transforms

def apply_augmentation(image, aug_pipeline):
    """
    Applies the given augmentation pipeline to an image.
    
    Parameters:
    - image: Input image (NumPy array).
    - aug_pipeline: Augmentation pipeline to apply.
    
    Returns:
    - Augmented image (NumPy array).
    """
    augmented = aug_pipeline(image=image)  # Apply augmentation
    return augmented["image"]  # Extract the augmented image

def visualize_augmentation(image_path, augmentation_name, augmentation_pipeline):
    """
    Visualizes original and augmented images side by side without using OpenCV.
    
    Parameters:
    - image_path: Path to the image file.
    - augmentation_name: The name of the applied augmentation.
    - augmentation_pipeline: Augmentation pipeline to apply.
    """
    # Load image using PIL
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)  # Convert to NumPy array
    
    # Apply augmentation
    augmented_image = apply_augmentation(image_np, augmentation_pipeline)
    
    # Convert back to PIL Image for visualization
    augmented_image = Image.fromarray(augmented_image) if isinstance(augmented_image, np.ndarray) else augmented_image

    # Plot the images
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    ax[1].imshow(augmented_image)
    ax[1].set_title(f"Augmented Image\n({augmentation_name})")  # Display augmentation type
    ax[1].axis("off")
    
    plt.show()

def run_test():
    """
    Run a test for Data
    
    Returns:
    - Augmented image (NumPy array).
    """
    augmentations = get_augmentation_pipelines()
    image_folder = "data/Sample-real/id0_0000"  # Folder containing frames

    # Get all image files from the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No images found in the folder!")
    else:
        # Pick a random image
        image_path = os.path.join(image_folder, random.choice(image_files))

        # Randomly select an augmentation variation
        augmentation_name, selected_augmentation = random.choice(augmentations)

        # Visualize augmentation
        visualize_augmentation(image_path, augmentation_name, selected_augmentation)

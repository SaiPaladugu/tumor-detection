# Install necessary libraries  for Google Colab
# !pip install ultralytics datasets pillow

import os
from datasets import load_dataset
from ultralytics import YOLO
import torch

# Paths
BASE_DIR = os.getcwd()  # Current working directory (useful for Google Colab)
OUTPUT_DIR = os.path.join(BASE_DIR, 'brain_tumor_yolo')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and clean the dataset
def load_and_clean_dataset():
    """
    Load, clean, and preprocess the dataset to YOLO-compatible format.
    """
    # Load the dataset
    dataset = load_dataset("mmenendezg/brain-tumor-object-detection")
    
    # Create YOLO-compatible directories
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

    # Process each split
    for split in ['train', 'validation', 'test']:
        process_split(dataset[split], split)

# Process each split of the dataset
def process_split(dataset_split, split_name):
    """
    Process a dataset split (train, validation, or test) and save images/labels in YOLO format.
    """
    images_dir = os.path.join(OUTPUT_DIR, split_name, 'images')
    labels_dir = os.path.join(OUTPUT_DIR, split_name, 'labels')

    for idx, example in enumerate(dataset_split):
        # Save the image
        image_path = os.path.join(images_dir, f"{idx:06d}.jpg")
        example['image'].save(image_path)

        # Save the labels (YOLO format)
        label_path = os.path.join(labels_dir, f"{idx:06d}.txt")
        with open(label_path, 'w') as label_file:
            for bbox, label in zip(example['objects']['bbox'], example['objects']['label']):
                x_center, y_center, width, height = normalize_bbox(bbox, example['image'].size)
                label_file.write(f"{label} {x_center} {y_center} {width} {height}\n")

# Normalize bounding boxes
def normalize_bbox(bbox, image_size):
    """
    Normalize bounding boxes to YOLO format.

    Args:
        bbox: List [x_min, y_min, width, height] (absolute pixel values).
        image_size: Tuple (width, height) of the image.

    Returns:
        Tuple (x_center, y_center, width, height) normalized to [0, 1].
    """
    image_width, image_height = image_size
    x_min, y_min, box_width, box_height = bbox
    x_center = (x_min + box_width / 2) / image_width
    y_center = (y_min + box_height / 2) / image_height
    box_width /= image_width
    box_height /= image_height
    return x_center, y_center, box_width, box_height

# Save YOLO data.yaml
def save_data_yaml(output_dir, num_classes):
    """
    Save a YOLO-compatible data.yaml file.
    """
    data_yaml = {
        "train": os.path.join(output_dir, "train", "images"),
        "val": os.path.join(output_dir, "validation", "images"),
        "test": os.path.join(output_dir, "test", "images"),  # Optional, used for explicit testing
        "nc": num_classes,
        "names": ["negative", "positive"]  # Adjust class names if necessary
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(f"train: {data_yaml['train']}\n")
        yaml_file.write(f"val: {data_yaml['val']}\n")
        yaml_file.write(f"nc: {data_yaml['nc']}\n")
        yaml_file.write("names:\n")
        for name in data_yaml["names"]:
            yaml_file.write(f"  - {name}\n")
    print(f"Saved YOLO data.yaml to {yaml_path}")

# Train YOLOv8
def train_yolo(data_yaml, model_type='yolov8n', epochs=3, img_size=640):
    """
    Train YOLOv8 on a custom dataset.

    Args:
        data_yaml (str): Path to the data.yaml file.
        model_type (str): Type of YOLOv8 model (e.g., 'yolov8n', 'yolov8s').
        epochs (int): Number of training epochs.
        img_size (int): Image size for training.

    Returns:
        YOLO: Trained YOLO model object.
    """
    # Initialize the model
    model = YOLO(f"{model_type}.pt")  # Load a pretrained YOLOv8 model

    # Train the model
    model.train(
        data=data_yaml,  # Path to dataset YAML file
        epochs=epochs,   # Number of epochs
        imgsz=img_size,  # Image size
        device="cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    )

    return model

# Evaluate YOLOv8
def evaluate_yolo(model, test_images_dir, data_yaml):
    """
    Evaluate a YOLOv8 model on a test split.

    Args:
        model (YOLO): Trained YOLO model object.
        test_images_dir (str): Path to the test images directory.
        data_yaml (str): Path to the data.yaml file.

    Returns:
        None
    """
    results = model.val(
        data=data_yaml,  # Path to dataset YAML file
        split="test"     # Explicitly specify the test split
    )
    print("Evaluation Results on Test Split:")
    print(results)

# Main pipeline
if __name__ == "__main__":
    # Step 1: Load and process dataset
    load_and_clean_dataset()

    # Step 2: Save data.yaml
    YAML_PATH = os.path.join(OUTPUT_DIR, "data.yaml")
    save_data_yaml(OUTPUT_DIR, num_classes=2)

    # Step 3: Train YOLOv8
    model = train_yolo(data_yaml=YAML_PATH, model_type='yolov8n', epochs=3, img_size=640)

    # Step 4: Evaluate YOLOv8 on test split
    test_images_dir = os.path.join(OUTPUT_DIR, "test", "images")
    evaluate_yolo(model, test_images_dir, YAML_PATH)

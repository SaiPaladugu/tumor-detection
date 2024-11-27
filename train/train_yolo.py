from ultralytics import YOLO
import os

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'brain_tumor_yolo')
YAML_PATH = os.path.join(DATA_DIR, 'data.yaml')  # Path to data.yaml

# Function to train YOLOv8
def train_yolo(data_yaml, model_type='yolov8n', epochs=50, img_size=640):
    """
    Train YOLOv8 on a custom dataset.

    Args:
        data_yaml (str): Path to the data.yaml file.
        model_type (str): Type of YOLOv8 model (e.g., 'yolov8n', 'yolov8s').
        epochs (int): Number of training epochs.
        img_size (int): Image size for training.

    Returns:
        None
    """
    # Initialize the model
    model = YOLO(f"{model_type}.pt")  # Load a pretrained YOLOv8 model

    # Train the model
    model.train(
        data=data_yaml,  # Path to dataset YAML file
        epochs=epochs,   # Number of epochs
        imgsz=img_size,  # Image size
        device="cpu"         # Use GPU (set to 'cpu' if no GPU is available)
    )

if __name__ == "__main__":
    # Check if data.yaml exists
    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(f"data.yaml not found at {YAML_PATH}")

    # Train YOLOv8
    train_yolo(data_yaml=YAML_PATH, model_type='yolov8n', epochs=3, img_size=640)

from ultralytics import YOLO
import os

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train', 'weights', 'best.pt')  # Path to trained YOLO model
INFERENCE_SOURCE = os.path.join(BASE_DIR, 'data', 'brain_tumor', 'valid', 'images')  # Path to input images or videos
OUTPUT_DIR = os.path.join(BASE_DIR, 'runs', 'inference')  # Output directory for results

# Function to run inference
def run_inference(model_path, source, output_dir, img_size=640, conf_threshold=0.25):
    """
    Run inference using a trained YOLOv8 model.

    Args:
        model_path (str): Path to the trained model file (e.g., 'best.pt').
        source (str): Path to the input source (image, video, or directory).
        output_dir (str): Directory to save the inference results.
        img_size (int): Image size for inference.
        conf_threshold (float): Confidence threshold for predictions.

    Returns:
        None
    """
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Run inference
    results = model.predict(
        source=source,     # Input source
        save=True,         # Save predictions
        save_txt=True,     # Save labels in YOLO format
        save_conf=True,    # Save confidence scores
        imgsz=img_size,    # Image size
        conf=conf_threshold,  # Confidence threshold
        project=output_dir,  # Directory for saving results
        name="inference",  # Subdirectory name for results
        device="cpu"           # Use GPU (set to 'cpu' if no GPU is available)
    )

    print(f"Inference completed. Results saved in {os.path.join(output_dir, 'inference')}")

if __name__ == "__main__":
    # Ensure the trained model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

    # Ensure the inference source exists
    if not os.path.exists(INFERENCE_SOURCE):
        raise FileNotFoundError(f"Inference source not found at {INFERENCE_SOURCE}")

    # Run inference
    run_inference(
        model_path=MODEL_PATH,
        source=INFERENCE_SOURCE,
        output_dir=OUTPUT_DIR,
        img_size=640,
        conf_threshold=0.25
    )

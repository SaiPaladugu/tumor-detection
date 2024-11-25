from PIL import Image, ImageDraw, ImageOps
import os

# Define the base directory for the dataset
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'brain_tumor')
ANNOTATED_DIR = os.path.join(BASE_DIR, 'data', 'annotated_images')  # Directory for annotated images

# Function to un-normalize YOLO labels
def unnormalize_label(label, image_width, image_height):
    """Convert normalized YOLO label to absolute bounding box coordinates."""
    class_id, x_center, y_center, width, height = map(float, label.split())
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    # Calculate absolute coordinates
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    return class_id, x_min, y_min, x_max, y_max

# Function to annotate images
def annotate_images(data_dir, annotated_dir):
    # Create the annotated images directory if it doesn't exist
    os.makedirs(annotated_dir, exist_ok=True)

    # Iterate over "train" and "valid" directories
    for split in ['train', 'valid']:
        split_dir = os.path.join(data_dir, split)
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        split_annotated_dir = os.path.join(annotated_dir, split)
        os.makedirs(split_annotated_dir, exist_ok=True)

        # Process image-label pairs
        for label_file in os.listdir(labels_dir):
            # Get the corresponding image file
            base_name = os.path.splitext(label_file)[0]
            image_path = os.path.join(images_dir, f"{base_name}.jpg")
            label_path = os.path.join(labels_dir, label_file)

            # Skip if the image file doesn't exist
            if not os.path.exists(image_path):
                print(f"Image not found for label: {label_path}")
                continue

            # Open the image
            with Image.open(image_path) as img:
                # Convert grayscale image to RGB for colored annotations
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                draw = ImageDraw.Draw(img)
                image_width, image_height = img.size

                # Read the label file and draw bounding boxes
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        class_id, x_min, y_min, x_max, y_max = unnormalize_label(line, image_width, image_height)
                        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                        draw.text((x_min, y_min), f"Class: {int(class_id)}", fill="yellow")

                # Save the annotated image
                annotated_image_path = os.path.join(split_annotated_dir, f"{base_name}_annotated.jpg")
                img.save(annotated_image_path)
                print(f"Annotated image saved: {annotated_image_path}")

# Run the script
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist.")
    
    annotate_images(DATA_DIR, ANNOTATED_DIR)

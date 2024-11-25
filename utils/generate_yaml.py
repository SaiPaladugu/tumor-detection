import os
import yaml

# Define the base directory for the dataset
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'brain_tumor')
YAML_PATH = os.path.join(DATA_DIR, 'data.yaml')  # Output YAML file

# Function to extract class names from the labels
def extract_class_names(labels_dir):
    class_ids = set()
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])  # Get the class ID
                    class_ids.add(class_id)
    return sorted(list(class_ids))  # Return sorted class IDs

# Function to generate the YAML file
def create_yaml(data_dir, yaml_path):
    # Directories for train and valid
    train_images_dir = os.path.join(data_dir, 'train', 'images')
    valid_images_dir = os.path.join(data_dir, 'valid', 'images')
    train_labels_dir = os.path.join(data_dir, 'train', 'labels')

    # Ensure directories exist
    if not os.path.exists(train_images_dir) or not os.path.exists(valid_images_dir):
        raise FileNotFoundError("Train or validation image directories not found!")

    # Extract class names (assume consecutive class IDs starting from 0)
    class_ids = extract_class_names(train_labels_dir)
    class_names = [f"class_{id}" for id in class_ids]  # Replace with actual names if available

    # Create YAML content
    data_yaml = {
        'train': train_images_dir,
        'val': valid_images_dir,
        'nc': len(class_names),
        'names': class_names,
    }

    # Write YAML file
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(data_yaml, yaml_file, default_flow_style=False)
    print(f"data.yaml created at {yaml_path}")

# Run the script
if __name__ == "__main__":
    create_yaml(DATA_DIR, YAML_PATH)

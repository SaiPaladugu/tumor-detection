import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from ultralytics import YOLO
import glob

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train', 'weights', 'best.pt')  # Path to trained YOLO model
OUTPUT_DIR = os.path.join(BASE_DIR, 'runs', 'inference')  # Output directory for results

import cv2

def run_inference(model_path, source, output_dir, img_size=640, conf_threshold=0.25):
    """
    Run inference using YOLO for classification and annotate the input image with the class label.
    """
    model = YOLO(model_path)
    results = model.predict(
        source=source,
        imgsz=img_size,
        conf=conf_threshold,
        device="cpu"
    )

    print("Results:", results)  # Debugging: Check predictions

    # Read the original image
    img = cv2.imread(source)
    if img is None:
        raise FileNotFoundError(f"Could not read the input image: {source}")

    # Process classification results
    if results is None or len(results) == 0:
        raise ValueError("No results were returned by the model. Check your model or input.")

    result = results[0]  # Get the first result
    if not hasattr(result, "probs") or result.probs is None:
        raise ValueError("No classification probabilities were found in the results.")

    # Get the top-1 class index and confidence
    class_index = result.probs.top1
    confidence = float(result.probs.top1conf)  # Convert confidence to float for formatting
    label = f"{model.names[class_index]} ({confidence:.2f})"

    print(f"Classification Result: {label}")  # Debugging: Log the classification result

    # Annotate the image with the class label
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_origin = (10, 30)  # Place the text at the top-left corner
    cv2.rectangle(img, (text_origin[0] - 5, text_origin[1] - text_size[1] - 5),
                  (text_origin[0] + text_size[0] + 5, text_origin[1] + 5), (0, 0, 255), -1)
    cv2.putText(img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save the annotated image
    output_subdirs = sorted(glob.glob(os.path.join(output_dir, "inference*")), key=os.path.getmtime)
    latest_output_dir = output_subdirs[-1] if output_subdirs else os.path.join(output_dir, "inference")
    os.makedirs(latest_output_dir, exist_ok=True)
    output_image_path = os.path.join(latest_output_dir, os.path.basename(source))
    cv2.imwrite(output_image_path, img)

    return output_image_path







# Function to handle file selection
def select_file():
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=(("Image Files", "*.jpg *.jpeg *.png"), ("All Files", "*.*"))
    )
    if file_path:
        entry_file_path.delete(0, tk.END)
        entry_file_path.insert(0, file_path)


# Function to execute inference
def on_inference():
    input_path = entry_file_path.get()
    if not os.path.exists(input_path):
        messagebox.showerror("Error", "Invalid file path. Please select a valid image file.")
        return

    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Error", "Model file not found. Ensure the model path is correct.")
        return

    try:
        # Run inference
        output_image_path = run_inference(MODEL_PATH, input_path, OUTPUT_DIR)

        # Display the result
        display_image(output_image_path)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Function to display the image
def display_image(image_path):
    image = Image.open(image_path)
    image.thumbnail((400, 400))  # Resize for display
    img_tk = ImageTk.PhotoImage(image)
    label_image.config(image=img_tk)
    label_image.image = img_tk


# Initialize GUI
root = tk.Tk()
root.title("YOLO Inference Tool")

# File selection
frame_file = tk.Frame(root)
frame_file.pack(pady=10)

label_file = tk.Label(frame_file, text="Select Image:")
label_file.pack(side=tk.LEFT, padx=5)

entry_file_path = tk.Entry(frame_file, width=50)
entry_file_path.pack(side=tk.LEFT, padx=5)

btn_browse = tk.Button(frame_file, text="Browse", command=select_file)
btn_browse.pack(side=tk.LEFT, padx=5)

# Inference button
btn_infer = tk.Button(root, text="Run Inference", command=on_inference)
btn_infer.pack(pady=10)

# Image display
label_image = tk.Label(root)
label_image.pack(pady=10)

# Start GUI
root.mainloop()

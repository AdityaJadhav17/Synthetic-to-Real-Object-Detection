from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml


# Function to predict and save images
def predict_and_save(model, image_path, output_path_txt):
    """
    Predict bounding boxes for an image and save them in YOLO format.
    
    Args:
        model: YOLOv8 model.
        image_path: Path to the input image.
        output_path_txt: Path to save the predictions.
    """

    # Perform prediction
    results = model.predict(image_path,conf=0.015)
    result = results[0]
    img_height, img_width = result.orig_shape

    # Save bounding boxes in YOLO format
    with open(output_path_txt, 'w') as f:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = box.conf.item()  
            # print(box.xywh[0][0].item(), box.xywh[0][1].item())
            x_center = box.xywh[0][0].item() / img_width
            y_center = box.xywh[0][1].item() / img_height
            # print(x_center, y_center)
            width = box.xywh[0][2].item() / img_width
            height = box.xywh[0][3].item() / img_height
            f.write(f"{cls_id} {conf} {x_center} {y_center} {width} {height}\n")

def main():
    # Set working directory
    this_dir = Path(__file__).parent
    os.chdir(this_dir)

    # Load test path from YAML
    with open(this_dir / 'yolo_params.yaml', 'r') as file:
        data = yaml.safe_load(file)
        if 'test' not in data or not data['test']:
            print("Add 'test: path/to/test/images' to yolo_params.yaml")
            exit()
        images_dir = Path(data['test'])
    
    # Validate test directory
    if not images_dir.exists():
        print(f"Test directory {images_dir} does not exist")
        exit()
    if not any(images_dir.glob('*')):
        print(f"Test directory {images_dir} is empty")
        exit()

    # Load the latest trained YOLO model
    detect_path = this_dir / "runs" / "detect"
    train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]
    if len(train_folders) == 0:
        raise ValueError("No training folders found")
    idx = 0
    if len(train_folders) > 1:
        choice = -1
        choices = list(range(len(train_folders)))
        while choice not in choices:
            print("Select the training folder:")
            for i, folder in enumerate(train_folders):
                print(f"{i}: {folder}")
            choice = input()
            if not choice.isdigit():
                choice = -1
            else:
                choice = int(choice)
        idx = choice

    model_path = detect_path / train_folders[idx] / "weights" / "best.pt"
    model = YOLO(model_path)

    # Directory with images to generate predictions
    output_dir = this_dir / "predictions" # Replace with the directory where you want to save predictions
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create labels subdirectories
    labels_output_dir = output_dir / 'labels'
    
    # images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through the images in the directory
    for img_path in images_dir.glob('*'):
        if img_path.suffix not in ['.png', '.jpg','.jpeg']:
            continue
        output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name  # Save label in 'labels' folder
        predict_and_save(model, img_path, output_path_txt)

    print(f"Bounding box labels saved in {labels_output_dir}")
    data = this_dir / 'yolo_params.yaml'
    print(f"Model parameters saved in {data}")

if __name__ == '__main__':
    main()
import cv2
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATASET_PATH = Path("dataset")
IMAGE_PATH = DATASET_PATH / "images"
ANNOTATION_PATH = DATASET_PATH / "annotations"
USED_CLASSES = ['A', 'B', 'C', 'D', 'E']

def parse_annotation(xml_file):
    """Parse Pascal VOC annotation XML file with dynamic handling of polygon points."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get filename
    filename = root.find('filename').text

    # Get image size
    size = root.find('size')
    if size is None:
        raise ValueError(f"Missing <size> element in annotation file: {xml_file}")
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # Get object information
    obj = root.find('object')
    if obj is None:
        raise ValueError(f"Missing <object> element in annotation file: {xml_file}")
    label = obj.find('name').text

    # Get polygon points
    polygon = obj.find('polygon')
    if polygon is None:
        raise ValueError(f"Missing <polygon> element in annotation file: {xml_file}")

    points = []

    # Extract all x and y pairs in the polygon
    for coord in polygon:
        if coord.tag.startswith('x'):  # Check if it's an x-coordinate
            coord_index = coord.tag[1:]  # Extract the index (e.g., '1' from 'x1')
            y_coord = polygon.find(f"y{coord_index}")  # Find the corresponding y-coordinate
            if y_coord is None:
                raise ValueError(f"Missing corresponding y{coord_index} for {coord.tag} in file: {xml_file}")

            try:
                x = round(float(coord.text))
                y = round(float(y_coord.text))
                points.append([x, y])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid numeric value for x{coord_index} or y{coord_index} in file: {xml_file}")

    if len(points) < 3:  # A valid polygon must have at least 3 points
        raise ValueError(f"Invalid polygon with less than 3 points in file: {xml_file}")

    return filename, np.array(points), label, (height, width)

def load_and_preprocess_dataset(image_size=(224, 224)):
    images = []
    masks = []
    labels = []
    
    xml_files = list(ANNOTATION_PATH.glob('*.xml'))
    print(f"Found {len(xml_files)} annotation files")
    images_files = list(IMAGE_PATH.glob('*.jpg'))
    print(f"Found {len(images_files)} image files")
    
    for xml_file in xml_files:
        try:
            filename, points, label, (height, width) = parse_annotation(xml_file)
            
            # Read and preprocess image
            img_path = IMAGE_PATH / filename
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create mask from polygon points
            mask = np.zeros((height, width), dtype=np.uint8)
            points = points.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
            
            # Resize image and mask
            image = cv2.resize(image, image_size)
            mask = cv2.resize(mask, image_size)
            
            # Normalize image to [0,1]
            image = image.astype(np.float32) / 255.0
            
            images.append(image)
            masks.append(mask)
            labels.append(label)
            
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")
    
    return np.array(images), np.array(masks), np.array(labels)

def display_dataset_info(images, masks, labels, classes):
    print("\nDataset Information:")
    print("-------------------")
    print(f"Total number of images: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Mask shape: {masks[0].shape}")
    print(f"Classes: {classes}")
    print("\nClass distribution:")
    for class_name in classes:
        count = np.sum(labels == class_name)
        print(f"{class_name}: {count}")

if __name__ == "__main__":
    # Load and preprocess dataset
    print("Loading and preprocessing dataset...")
    images, masks, labels = load_and_preprocess_dataset()
    
    # Display dataset information
    display_dataset_info(images, masks, labels, USED_CLASSES)
    
    # One-hot encode the labels
    lb = LabelBinarizer()
    labels_one_hot = lb.fit_transform(labels)
    
    # Split dataset
    print("\nSplitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, masks, 
        train_size=0.7, 
        random_state=42
    )
    
    # Split remaining data into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=0.5,  # This gives 15% each for val and test
        random_state=42
    )
    
    print("\nDataset splits:")
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Visualize examples
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(2, 3, i+1)
        plt.imshow(X_train[i])
        plt.title(f"Image {i+1}")
        plt.axis('off')
        
        plt.subplot(2, 3, i+4)
        plt.imshow(y_train[i], cmap='gray')
        plt.title(f"Mask {i+1}")
        plt.axis('off')
    plt.tight_layout()
    
    # Draw data distribution
    class_counts = {}
    for label in labels:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    
    class_names = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(8, 6))
    plt.bar(class_names, counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Classes in Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
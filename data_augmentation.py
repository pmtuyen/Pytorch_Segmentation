import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm

class DataAugmenter:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / 'images'
        self.annotation_dir = self.dataset_dir / 'annotations'
        
        if not self.image_dir.exists() or not self.annotation_dir.exists():
            raise ValueError("Dataset directory must contain 'images' and 'annotations' subdirectories")

    def rotate_points(self, points, angle, center, image_shape):
        """Rotate polygon points around center"""
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        points = np.array(points, dtype=np.float32)
        points = points.reshape(-1, 2)
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])
        transformed_points = M.dot(points_ones.T).T
        # Ensure points are within image bounds
        transformed_points[:, 0] = np.clip(transformed_points[:, 0], 0, image_shape[1]-1)
        transformed_points[:, 1] = np.clip(transformed_points[:, 1], 0, image_shape[0]-1)
        return transformed_points

    def flip_points(self, points, image_shape, flip_code):
        """Flip polygon points horizontally or vertically"""
        points = np.array(points)
        if flip_code == 1:  # horizontal flip
            points[:, 0] = image_shape[1] - points[:, 0]
        elif flip_code == 0:  # vertical flip
            points[:, 1] = image_shape[0] - points[:, 1]
        return points

    def augment_data(self):
        """Perform data augmentation"""
        xml_files = list(self.annotation_dir.glob('*.xml'))
        
        for xml_file in tqdm(xml_files, desc="Augmenting data"):
            # Skip if file is already augmented
            if any(suffix in xml_file.stem for suffix in ['rotate_90', 'rotate_270', 'flip_h', 'flip_v']):
                continue
                
            # Parse original annotation
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image filename and load image
            filename = root.find('filename').text
            image_path = self.image_dir / filename
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not read image: {image_path}")
                continue
            
            # Get polygon points from object
            obj = root.find('.//object')
            if obj is None:
                print(f"No object element found in {xml_file}")
                continue
                
            polygon = obj.find('.//polygon')
            if polygon is None:
                print(f"No polygon found in {xml_file}")
                continue
            
            # Extract letter class
            class_name = obj.find('name').text
            
            # Get points
            points = []
            for i in range(1, len(polygon)):
                if polygon[i].tag.startswith('x'):
                    x = float(polygon[i].text)
                    y = float(polygon[i+1].text)
                    points.append([x, y])
            points = np.array(points)
            
            # Get image dimensions
            height, width = image.shape[:2]
            center = (width/2, height/2)
            
            # Define augmentations
            augmentations = [
                ('rotate_90', lambda img, pts: (
                    cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
                    self.rotate_points(pts, -90, center, img.shape)
                )),
                ('rotate_270', lambda img, pts: (
                    cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
                    self.rotate_points(pts, 90, center, img.shape)
                )),
                ('flip_h', lambda img, pts: (
                    cv2.flip(img, 1),
                    self.flip_points(pts, (height, width), 1)
                )),
                ('flip_v', lambda img, pts: (
                    cv2.flip(img, 0),
                    self.flip_points(pts, (height, width), 0)
                ))
            ]
            
            # Apply augmentations
            for aug_name, aug_func in augmentations:
                new_base = f"{xml_file.stem}_{aug_name}"
                new_image_name = f"{new_base}.jpg"
                new_xml_name = f"{new_base}.xml"
                
                # Skip if augmented file already exists
                if (self.image_dir / new_image_name).exists():
                    continue
                
                # Apply transformation
                aug_image, aug_points = aug_func(image.copy(), points.copy())
                
                # Save augmented image
                cv2.imwrite(str(self.image_dir / new_image_name), aug_image)
                
                # Create new XML
                new_root = ET.Element('annotation')
                ET.SubElement(new_root, 'filename').text = new_image_name
                
                # Add size information
                size = ET.SubElement(new_root, 'size')
                ET.SubElement(size, 'width').text = str(aug_image.shape[1])
                ET.SubElement(size, 'height').text = str(aug_image.shape[0])
                
                # Create object element with polygon
                obj = ET.SubElement(new_root, 'object')
                ET.SubElement(obj, 'name').text = class_name
                
                # Add polygon points within object
                polygon = ET.SubElement(obj, 'polygon')
                for i, (x, y) in enumerate(aug_points):
                    ET.SubElement(polygon, f'x{i+1}').text = str(int(round(x)))
                    ET.SubElement(polygon, f'y{i+1}').text = str(int(round(y)))
                
                # Save augmented annotation
                tree = ET.ElementTree(new_root)
                tree.write(str(self.annotation_dir / new_xml_name))

if __name__ == "__main__":
    DATASET_DIR = Path("dataset")
    
    print(f"Starting data augmentation...")
    print(f"Using dataset directory: {DATASET_DIR}")
    
    augmenter = DataAugmenter(DATASET_DIR)
    augmenter.augment_data()
    
    # Print summary
    xml_files = list(augmenter.annotation_dir.glob('*.xml'))
    original_files = [f for f in xml_files if not any(
        suffix in f.stem for suffix in ['rotate_90', 'rotate_270', 'flip_h', 'flip_v']
    )]
    
    print(f"\nAugmentation complete!")
    print(f"Original images: {len(original_files)}")
    print(f"Augmented images added: {len(xml_files) - len(original_files)}")
    print(f"Total images: {len(xml_files)}")
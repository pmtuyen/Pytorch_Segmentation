from pathlib import Path
import os

def clean_augmented_data(dataset_dir):
    """Remove all augmented images and their annotations"""
    # Augmentation suffixes to look for
    aug_suffixes = ['rotate_90', 'rotate_270', 'flip_h', 'flip_v']
    
    # Get paths
    dataset_dir = Path(dataset_dir)
    image_dir = dataset_dir / 'images'
    annotation_dir = dataset_dir / 'annotations'
    
    # Verify directories exist
    if not image_dir.exists() or not annotation_dir.exists():
        print("Error: Dataset directory must contain 'images' and 'annotations' subdirectories")
        return
    
    # Counter for deleted files
    deleted_count = 0
    
    # Remove augmented files
    for suffix in aug_suffixes:
        # Remove images
        for img_file in image_dir.glob(f'*_{suffix}.*'):
            try:
                os.remove(img_file)
                deleted_count += 1
                print(f"Deleted image: {img_file.name}")
            except Exception as e:
                print(f"Error deleting {img_file}: {e}")
        
        # Remove annotations
        for xml_file in annotation_dir.glob(f'*_{suffix}.xml'):
            try:
                os.remove(xml_file)
                deleted_count += 1
                print(f"Deleted annotation: {xml_file.name}")
            except Exception as e:
                print(f"Error deleting {xml_file}: {e}")
    
    print(f"\nCleanup complete!")
    print(f"Total files deleted: {deleted_count}")
    print(f"Remaining images: {len(list(image_dir.glob('*.*')))}")
    print(f"Remaining annotations: {len(list(annotation_dir.glob('*.xml')))}")

if __name__ == "__main__":
    # Path to your dataset directory
    DATASET_DIR = Path("dataset")
    
    # Confirm before deletion
    print("This will delete all augmented data files.")
    confirmation = input("Do you want to continue? (y/n): ")
    
    if confirmation.lower() == 'y':
        clean_augmented_data(DATASET_DIR)
    else:
        print("Operation cancelled.")
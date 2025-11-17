"""
Hand Sign Recognition - Grayscale Dataset Converter
Converts color dataset to grayscale with CLAHE enhancement
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm


def convert_to_grayscale(source_dir, dest_dir, apply_clahe=True):
    """
    Convert entire dataset from color to grayscale
    
    Args:
        source_dir: Path to original colored dataset
        dest_dir: Path to save grayscale dataset
        apply_clahe: Apply CLAHE for better contrast and lighting robustness
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Create CLAHE object
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    total_images = 0
    converted_images = 0
    failed_images = []
    
    print("\n" + "=" * 70)
    print("🎨 GRAYSCALE DATASET CONVERTER")
    print("=" * 70)
    print(f"📁 Source: {source_path}")
    print(f"📁 Destination: {dest_path}")
    print(f"🔧 CLAHE Enhancement: {'Enabled' if apply_clahe else 'Disabled'}")
    print("=" * 70)
    
    # Process train and test directories
    for split in ['train', 'test']:
        split_source = source_path / split
        split_dest = dest_path / split
        
        if not split_source.exists():
            print(f"\n⚠️  {split_source} does not exist, skipping...")
            continue
        
        # Get all class folders
        class_folders = sorted([f for f in split_source.iterdir() if f.is_dir()])
        
        if not class_folders:
            print(f"\n⚠️  No class folders found in {split_source}")
            continue
        
        print(f"\n📂 Processing {split.upper()} dataset ({len(class_folders)} classes)...")
        
        for class_folder in tqdm(class_folders, desc=f"Converting {split}"):
            class_name = class_folder.name
            
            # Create destination folder
            dest_class_folder = split_dest / class_name
            dest_class_folder.mkdir(parents=True, exist_ok=True)
            
            # Get all images
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(class_folder.glob(ext))
                image_files.extend(class_folder.glob(ext.upper()))
            
            total_images += len(image_files)
            
            for img_path in image_files:
                try:
                    # Read image
                    img = cv2.imread(str(img_path))
                    
                    if img is None:
                        failed_images.append(str(img_path))
                        continue
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Apply CLAHE
                    if apply_clahe:
                        gray = clahe.apply(gray)
                    
                    # Save with same filename
                    dest_img_path = dest_class_folder / img_path.name
                    cv2.imwrite(str(dest_img_path), gray)
                    
                    converted_images += 1
                    
                except Exception as e:
                    failed_images.append(f"{img_path} (Error: {e})")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Total images found: {total_images}")
    print(f"Successfully converted: {converted_images}")
    print(f"Failed: {len(failed_images)}")
    
    if failed_images:
        print(f"\n❌ Failed images:")
        for failed in failed_images[:10]:  # Show first 10
            print(f"  - {failed}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
    
    print(f"\n💾 Grayscale dataset saved to: {dest_path}")
    print("=" * 70)


def verify_conversion(original_dir, grayscale_dir):
    """
    Verify that conversion was successful by comparing image counts
    
    Args:
        original_dir: Original dataset directory
        grayscale_dir: Grayscale dataset directory
    """
    original_path = Path(original_dir)
    grayscale_path = Path(grayscale_dir)
    
    print("\n" + "=" * 70)
    print("🔍 VERIFYING CONVERSION")
    print("=" * 70)
    
    for split in ['train', 'test']:
        orig_split = original_path / split
        gray_split = grayscale_path / split
        
        if not orig_split.exists():
            continue
        
        print(f"\n📁 Checking {split.upper()} set...")
        
        orig_classes = sorted([f for f in orig_split.iterdir() if f.is_dir()])
        gray_classes = sorted([f for f in gray_split.iterdir() if f.is_dir()])
        
        if len(orig_classes) != len(gray_classes):
            print(f"  ⚠️  Class count mismatch!")
            print(f"     Original: {len(orig_classes)} | Grayscale: {len(gray_classes)}")
        else:
            print(f"  ✅ Class count matches: {len(orig_classes)}")
        
        total_orig = 0
        total_gray = 0
        
        for class_folder in orig_classes:
            class_name = class_folder.name
            gray_class = gray_split / class_name
            
            # Count images
            orig_count = len(list(class_folder.glob('*.jpg')) + 
                           list(class_folder.glob('*.jpeg')) + 
                           list(class_folder.glob('*.png')) +
                           list(class_folder.glob('*.bmp')))
            
            gray_count = 0
            if gray_class.exists():
                gray_count = len(list(gray_class.glob('*.jpg')) + 
                               list(gray_class.glob('*.jpeg')) + 
                               list(gray_class.glob('*.png')) +
                               list(gray_class.glob('*.bmp')))
            
            total_orig += orig_count
            total_gray += gray_count
            
            if orig_count != gray_count:
                print(f"  ⚠️  Mismatch in class '{class_name}': {orig_count} -> {gray_count}")
        
        print(f"\n  Total images:")
        print(f"    Original: {total_orig}")
        print(f"    Grayscale: {total_gray}")
        
        if total_orig == total_gray:
            print(f"  ✅ All images converted successfully!")
        else:
            print(f"  ⚠️  Image count mismatch!")
    
    print("\n" + "=" * 70)


# ==================== MAIN ====================

if __name__ == "__main__":
    # Configuration
    BASE_DIR = Path(__file__).resolve().parent.parent
    SOURCE_DIR = BASE_DIR / 'data'
    DEST_DIR = BASE_DIR / 'data_grayscale'
    
    print("\n" + "=" * 70)
    print("🎨 HAND SIGN DATASET - GRAYSCALE CONVERTER")
    print("=" * 70)
    
    # Check if source exists
    if not SOURCE_DIR.exists():
        print(f"\n❌ Source directory not found: {SOURCE_DIR}")
        print("Please ensure your dataset is in the 'data' folder")
        exit(1)
    
    # Ask for confirmation
    print(f"\nThis will convert all images from:")
    print(f"  {SOURCE_DIR}")
    print(f"To grayscale and save in:")
    print(f"  {DEST_DIR}")
    
    response = input("\nProceed? (y/n): ")
    
    if response.lower() != 'y':
        print("❌ Conversion cancelled")
        exit(0)
    
    # Convert dataset
    convert_to_grayscale(
        source_dir=SOURCE_DIR,
        dest_dir=DEST_DIR,
        apply_clahe=True
    )
    
    # Verify conversion
    verify_response = input("\nVerify conversion? (y/n): ")
    if verify_response.lower() == 'y':
        verify_conversion(SOURCE_DIR, DEST_DIR)
    
    print("\n✅ Done! You can now use the grayscale dataset for training.")
    print("=" * 70)
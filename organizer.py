"""
Photo Organizer - Automatically organize homework-related photos.
Scans a directory of photos, classifies them, and moves homework-related images
to a separate folder.
"""

import os
import shutil
from pathlib import Path
from classifier import HomeworkImageClassifier


class PhotoOrganizer:
    """Organizes photos by moving homework-related images to a separate folder."""
    
    def __init__(self, source_dir, homework_dir="school_photos"):
        """
        Initialize the photo organizer.
        
        Args:
            source_dir: Directory containing photos to organize
            homework_dir: Directory where homework photos will be moved (default: "school_photos")
        """
        self.source_dir = Path(source_dir)
        self.homework_dir = Path(homework_dir)
        self.classifier = HomeworkImageClassifier()
        
        # Supported image formats
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
    def create_homework_folder(self):
        """Create the homework folder if it doesn't exist."""
        self.homework_dir.mkdir(parents=True, exist_ok=True)
        print(f"Homework folder: {self.homework_dir.absolute()}")
        
    def get_image_files(self):
        """
        Get all image files from the source directory.
        
        Returns:
            List of Path objects for image files
        """
        image_files = []
        for file_path in self.source_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                image_files.append(file_path)
        return image_files
    
    def organize_photos(self, dry_run=False):
        """
        Organize photos by moving homework-related images.
        
        Args:
            dry_run: If True, only simulate the organization without moving files
            
        Returns:
            Dictionary with statistics about the organization
        """
        if not self.source_dir.exists():
            print(f"Error: Source directory '{self.source_dir}' does not exist!")
            return None
        
        # Create homework folder
        self.create_homework_folder()
        
        # Get all image files
        image_files = self.get_image_files()
        print(f"\nFound {len(image_files)} images to process...")
        
        if not image_files:
            print("No images found to organize.")
            return {"total": 0, "homework": 0, "regular": 0}
        
        # Classify images
        print("\nClassifying images...")
        results = self.classifier.classify_batch([str(f) for f in image_files])
        
        # Organize based on classification
        stats = {"total": len(results), "homework": 0, "regular": 0, "errors": 0}
        
        print("\n" + "="*60)
        print("ORGANIZATION RESULTS")
        print("="*60)
        
        for img_path, is_homework, confidence, prediction in results:
            img_file = Path(img_path)
            
            if is_homework:
                stats["homework"] += 1
                dest_path = self.homework_dir / img_file.name
                
                if dry_run:
                    print(f"[DRY RUN] Would move: {img_file.name} → school_photos/")
                    print(f"  Reason: {prediction} ({confidence:.2%} confidence)")
                else:
                    try:
                        shutil.move(str(img_file), str(dest_path))
                        print(f"✓ Moved: {img_file.name} → school_photos/")
                        print(f"  Reason: {prediction} ({confidence:.2%} confidence)")
                    except Exception as e:
                        print(f"✗ Error moving {img_file.name}: {e}")
                        stats["errors"] += 1
            else:
                stats["regular"] += 1
                print(f"○ Kept: {img_file.name} (classified as: {prediction})")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total images processed: {stats['total']}")
        print(f"Homework images {'would be moved' if dry_run else 'moved'}: {stats['homework']}")
        print(f"Regular photos kept: {stats['regular']}")
        if stats['errors'] > 0:
            print(f"Errors encountered: {stats['errors']}")
        print("="*60)
        
        return stats


def main():
    """Main function to run the photo organizer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Organize photos by detecting and separating homework-related images"
    )
    parser.add_argument(
        "source_dir",
        help="Directory containing photos to organize"
    )
    parser.add_argument(
        "--homework-dir",
        default="school_photos",
        help="Directory where homework photos will be moved (default: school_photos)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate organization without actually moving files"
    )
    
    args = parser.parse_args()
    
    # Create organizer and run
    organizer = PhotoOrganizer(args.source_dir, args.homework_dir)
    organizer.organize_photos(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

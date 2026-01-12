"""
Image classifier using EfficientNet to detect homework-related images.
Classifies images as homework-related (textbooks, whiteboards, notes) or regular photos.
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


class HomeworkImageClassifier:
    """Classifier to detect homework-related images using EfficientNet."""
    
    def __init__(self):
        """Initialize the EfficientNet model with ImageNet weights."""
        print("Loading EfficientNet model...")
        self.model = EfficientNetB0(weights='imagenet', include_top=True)
        print("Model loaded successfully!")
        
        # ImageNet classes that are typically homework-related
        # These are indices in the ImageNet classes
        self.homework_keywords = [
            'notebook', 'book', 'paper', 'pen', 'pencil', 'desk',
            'monitor', 'screen', 'keyboard', 'laptop', 'computer',
            'whiteboard', 'blackboard', 'chalkboard', 'book_jacket',
            'pencil_box', 'binder', 'calculator'
        ]
        
    def load_and_preprocess_image(self, img_path, target_size=(224, 224)):
        """
        Load and preprocess an image for EfficientNet.
        
        Args:
            img_path: Path to the image file
            target_size: Target size for the image (default: 224x224 for EfficientNet)
            
        Returns:
            Preprocessed image array
        """
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def is_homework_related(self, img_path, threshold=0.3):
        """
        Determine if an image is homework-related.
        
        Args:
            img_path: Path to the image file
            threshold: Confidence threshold for classification (default: 0.3)
            
        Returns:
            Tuple of (is_homework: bool, confidence: float, top_prediction: str)
        """
        # Preprocess image
        preprocessed_img = self.load_and_preprocess_image(img_path)
        
        # Get predictions
        predictions = self.model.predict(preprocessed_img, verbose=0)
        
        # Decode predictions to get readable class names
        decoded_predictions = tf.keras.applications.efficientnet.decode_predictions(
            predictions, top=5
        )[0]
        
        # Check if any of the top predictions match homework keywords
        for _, class_name, confidence in decoded_predictions:
            class_name_lower = class_name.lower()
            for keyword in self.homework_keywords:
                if keyword in class_name_lower:
                    return True, float(confidence), class_name
        
        # Additional heuristic: check for document-like characteristics
        # High contrast, rectangular shapes, text-like patterns
        try:
            img = Image.open(img_path)
            img_gray = img.convert('L')
            img_array = np.array(img_gray)
            
            # Calculate variance (documents tend to have high variance)
            variance = np.var(img_array)
            
            # If variance is very high, it might be a document/whiteboard
            if variance > 5000:
                return True, 0.5, "document_heuristic"
        except Exception:
            pass
        
        # Get top prediction for logging
        top_class = decoded_predictions[0][1]
        top_confidence = float(decoded_predictions[0][2])
        
        return False, top_confidence, top_class
    
    def classify_batch(self, image_paths):
        """
        Classify multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of tuples (image_path, is_homework, confidence, prediction)
        """
        results = []
        for img_path in image_paths:
            try:
                is_hw, conf, pred = self.is_homework_related(img_path)
                results.append((img_path, is_hw, conf, pred))
                print(f"✓ Processed: {img_path}")
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
                results.append((img_path, False, 0.0, "error"))
        return results


def main():
    """Demo function to test the classifier."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python classifier.py <image_path>")
        print("Example: python classifier.py photo.jpg")
        sys.exit(1)
    
    img_path = sys.argv[1]
    
    # Initialize classifier
    classifier = HomeworkImageClassifier()
    
    # Classify image
    is_homework, confidence, prediction = classifier.is_homework_related(img_path)
    
    print("\n" + "="*50)
    print(f"Image: {img_path}")
    print(f"Homework-related: {'YES' if is_homework else 'NO'}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Top prediction: {prediction}")
    print("="*50)


if __name__ == "__main__":
    main()

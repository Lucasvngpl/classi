"""
Configuration settings for Photo Organizer.
Modify these values to customize the behavior.
"""

# Classification settings
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to classify as homework
BATCH_SIZE = 10  # Number of images to process at once

# Homework-related keywords (ImageNet classes)
HOMEWORK_KEYWORDS = [
    'notebook',
    'book',
    'paper',
    'pen',
    'pencil',
    'desk',
    'monitor',
    'screen',
    'keyboard',
    'laptop',
    'computer',
    'whiteboard',
    'blackboard',
    'chalkboard',
    'book_jacket',
    'pencil_box',
    'binder',
    'calculator',
    'web_site',  # Screenshots of educational sites
    'menu',  # Often captures written content
]

# Image processing settings
IMAGE_SIZE = (224, 224)  # EfficientNetB0 input size
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

# Organization settings
DEFAULT_HOMEWORK_FOLDER = "school_photos"
MOVE_FILES = True  # If False, copy instead of move

# Document detection heuristics
VARIANCE_THRESHOLD = 5000  # For detecting high-contrast documents

# Model settings
MODEL_NAME = "EfficientNetB0"  # Can be changed to other EfficientNet variants
USE_IMAGENET_WEIGHTS = True

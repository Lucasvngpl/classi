# Photo Organizer 📚📱

Automatically classify and organize photos by detecting homework-related images (textbooks, whiteboards, notes) using EfficientNet deep learning model.

## Overview

This is a minimal hackathon-ready template that uses AI to:
- 🧠 **Classify images** using EfficientNet to detect school/homework content
- 📁 **Auto-organize** by moving homework photos to a separate folder
- 🚀 **Fast setup** with just Python and TensorFlow

Perfect for keeping your photo gallery clean by automatically separating study materials from personal photos!

## Features

- ✅ EfficientNet-based image classification
- ✅ Detects textbooks, whiteboards, notes, and study materials
- ✅ Automatic folder organization
- ✅ Dry-run mode to preview changes
- ✅ Batch processing support
- ✅ Easy to extend and customize

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Lucasvngpl/photo-organizer.git
   cd photo-organizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - TensorFlow (for EfficientNet model)
   - Pillow (for image processing)
   - NumPy (for numerical operations)

### Usage

#### Organize Photos

```bash
python organizer.py /path/to/your/photos
```

This will:
1. Scan all images in the specified directory
2. Classify each image using EfficientNet
3. Move homework-related images to a `school_photos/` folder
4. Keep regular photos in the original location

#### Dry Run (Preview Mode)

Test without moving any files:

```bash
python organizer.py /path/to/your/photos --dry-run
```

#### Custom Output Folder

Specify a custom folder for homework photos:

```bash
python organizer.py /path/to/your/photos --homework-dir my_study_folder
```

#### Test Single Image

Classify a single image:

```bash
python classifier.py photo.jpg
```

## Example Output

```
Found 15 images to process...

Classifying images...
✓ Processed: textbook_page.jpg
✓ Processed: vacation_pic.jpg
✓ Processed: whiteboard_notes.jpg

============================================================
ORGANIZATION RESULTS
============================================================
✓ Moved: textbook_page.jpg → school_photos/
  Reason: book_jacket (0.85 confidence)
○ Kept: vacation_pic.jpg (classified as: seashore)
✓ Moved: whiteboard_notes.jpg → school_photos/
  Reason: whiteboard (0.72 confidence)

============================================================
SUMMARY
============================================================
Total images processed: 15
Homework images moved: 3
Regular photos kept: 12
============================================================
```

## How It Works

### Classification Model

The app uses **EfficientNetB0**, a state-of-the-art convolutional neural network pre-trained on ImageNet. It detects:

- 📖 Books and textbooks
- 📝 Notebooks and papers
- 🖥️ Computer screens and laptops
- ⬜ Whiteboards and blackboards
- ✏️ Pens, pencils, and stationery
- 💻 Study-related objects

### Classification Logic

1. **Pre-trained Model**: Uses EfficientNet with ImageNet weights
2. **Keyword Matching**: Checks if predictions match homework-related keywords
3. **Heuristic Analysis**: Analyzes image characteristics (contrast, variance) to detect documents
4. **Confidence Threshold**: Only moves images with sufficient confidence

## Customization

### Adding More Keywords

Edit `classifier.py` to add more homework-related keywords:

```python
self.homework_keywords = [
    'notebook', 'book', 'paper', 'pen', 'pencil',
    'your_custom_keyword_here'
]
```

### Adjusting Confidence Threshold

Modify the threshold in `classifier.py`:

```python
def is_homework_related(self, img_path, threshold=0.3):  # Change 0.3 to your desired value
```

### Supported Image Formats

- JPG/JPEG
- PNG
- BMP
- GIF
- WebP

## Android Adaptation

While this is currently a Python implementation, it can be adapted for Android:

1. **Convert to TensorFlow Lite**: Use TFLite for mobile deployment
2. **Android ML Kit**: Integrate with Android's ML Kit
3. **Background Service**: Run classification as a background service
4. **Media Store API**: Use Android's MediaStore to organize photos
5. **Custom Album**: Create hidden album or use `.nomedia` file

See [Android ML Kit documentation](https://developers.google.com/ml-kit) for mobile integration.

## Project Structure

```
photo-organizer/
├── classifier.py          # EfficientNet-based image classifier
├── organizer.py          # Main photo organization logic
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Troubleshooting

### Issue: Model download fails
**Solution**: Ensure you have internet connection. TensorFlow will download EfficientNet weights on first run.

### Issue: Out of memory
**Solution**: Process images in smaller batches or use a smaller model variant.

### Issue: Slow processing
**Solution**: The first run downloads the model (~30MB). Subsequent runs are faster. Consider using GPU acceleration.

## Performance

- **Model Size**: ~30MB (EfficientNetB0)
- **Speed**: ~1-2 seconds per image on CPU
- **Accuracy**: Good for obvious homework content (books, whiteboards)

## Future Enhancements

- [ ] Fine-tune model on homework-specific dataset
- [ ] Add GUI interface
- [ ] Support for cloud storage (Google Drive, Dropbox)
- [ ] Mobile app (Android/iOS)
- [ ] OCR integration for text detection
- [ ] Duplicate detection
- [ ] Smart album creation

## License

MIT License - feel free to use for your hackathon or personal projects!

## Contributing

Contributions welcome! This is a hackathon template, so keep it simple and focused.

## Credits

Built with:
- [TensorFlow](https://www.tensorflow.org/) & [EfficientNet](https://arxiv.org/abs/1905.11946)
- [Keras](https://keras.io/)
- Python & Love ❤️
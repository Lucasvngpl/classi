import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

BASE_PATH = "../data/"
IMG_SIZE = 512
BATCH_SIZE = 32
AUGMENT = True

def build_dataset():
    if AUGMENT:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.8, 1.2],
            zoom_range=0.2,
            fill_mode="nearest"
        )

    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    # No augmentation for validation
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_set = train_datagen.flow_from_directory(
        os.path.join(BASE_PATH, ""),
    )
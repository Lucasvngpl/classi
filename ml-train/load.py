from PIL import Image
from tensorflow import keras
import tensorflow as tf
import os
from model_params import *

def create_augmentation():
    return keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0))  # Clip
    ], name="augmentation")

def load_all(fp="../dataset-full"):

    # Load train set
    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(fp, "train"),
        labels="inferred",
        label_mode="binary",
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=True,
        seed=67
    )

    # Load valid set
    valid_ds = keras.utils.image_dataset_from_directory(
        os.path.join(fp, "valid"),
        labels="inferred",
        label_mode="binary",
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=False
    )

    # Load test set
    test_ds = keras.utils.image_dataset_from_directory(
        os.path.join(fp, "test"),
        labels="inferred",
        label_mode="binary",
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"\nClasses found: {class_names}")

    n_train = sum(1 for _ in train_ds.unbatch())
    n_valid = sum(1 for _ in valid_ds.unbatch())
    n_test = sum(1 for _ in test_ds.unbatch())

    print(f"\nTrain: {n_train} images")
    print(f"Validation: {n_valid} images")
    print(f"Test: {n_test} images")

    # Set norm, clean 255
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    aug = create_augmentation()

    train_ds = train_ds.map(
        lambda x, y: (aug(norm(x), training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    valid_ds = valid_ds.map(
        lambda x, y: (norm(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    test_ds = test_ds.map(
        lambda x, y: (norm(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Performance stuff
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    load_all()
import tensorflow as tf
import warnings
from tensorflow.keras.applications import EfficientNetB0
from model_params import *


class Conv:
    def __init__(self):
        if self.debug_gpu():
            print("Using GPU")
        else:
            warnings.warn("GPU not found. Defaulting to CPU\nModel should still train but may be slow", UserWarning)

        self.model = self.new_model()

    # Debug GPU info | True if GPU installed
    def debug_gpu(self):
        devices = tf.config.list_physical_devices()
        print("\nDevices: ", devices)

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            details = tf.config.experimental.get_device_details(gpus[0])
            print("GPU details: ", details)

        return gpus

    def new_model(self, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
        
        # Raw CNN
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.BatchNormalization(),

            # Second conv block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.BatchNormalization(),

            # Third conv block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.BatchNormalization(),

            # Fourth conv block
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.BatchNormalization(),

            # Flatten and dense layers
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        return model

    def compile_model(self, lr=1e-3): # tells keras how to train the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr), # Adam optimizer with learning rate, updates weights based on loss
            loss="binary_crossentropy", # loss function for binary classification
            metrics=["accuracy"] # track accuracy during training   
        )
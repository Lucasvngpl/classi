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

        # Load pre-trained EfficientNetB0
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,      # Remove ImageNet classifier so it replaces the original classifier classes with our own
            weights="imagenet",     # Use pre-trained weights
            input_shape=input_shape,
            pooling=None           # Pooling later
        )
        
        # Freeze base model, read more on this
        base_model.trainable = False
        
        # Build: EfficientNet → Dense(128) → Dense(1)
        model = tf.keras.models.Sequential([
            base_model,
            tf.keras.GlobalAveragePooling2D(), # Ensure output is flat vector
            tf.keras.layers.Dropout(0.3), 
            tf.keras.layers.Dense(128, activation="relu"), # Newly added dense layer with 128 units
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid") # Final output layer for binary classification spits out probability
        ])
    

        return model
    def compile_model(self): # tells keras how to train the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Adam optimizer with learning rate, updates weights based on loss
            loss="binary_crossentropy", # loss function for binary classification
            metrics=["accuracy"] # track accuracy during training   
        )
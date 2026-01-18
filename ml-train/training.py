import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from model import Conv
from load import load_all
import os
import datetime
import ssl
from model_params import *

ssl._create_default_https_context = ssl._create_unverified_context

class Trainer:
    def __init__(self):
        self.conv = Conv()
        self.conv.compile_model()

        self.train_data, self.valid_data, self.test_data = load_all()

        os.makedirs("../models/checkpoints", exist_ok=True)
        os.makedirs("../models/final", exist_ok=True)
    
    def get_callbacks(self):
        # Early stopping callback
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",      # Monitor validation loss
                patience=5,              # Stop after 5 epochs of no improvement
                restore_best_weights=True,  # Restore model weights from the epoch with the best validation loss
                verbose = 1
            ),

            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"../models/checkpoints/model_epoch_{{epoch:02d}}_val_acc_{{val_accuracy:.4f}}.h5",
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]

        return callbacks

    def train(self):
        # check if loaded
        if self.train_data is None or self.valid_data is None or self.test_data is None:
            raise ValueError("Datasets not loaded. Please load datasets before training.")
        
        # Train the model
        # fit() updates model weights using TRAINING data
        # validation_data is used ONLY for monitoring (no weight updates)
        self.history = self.conv.model.fit( # Saves the training results returned by fit() into self.history.
            self.train_data,                       # Training data - model learns from this
            epochs=EPOCHS,                    # HYPERPARAMETER: Number of full passes through data
            validation_data=self.valid_data,       # Validation data - detects overfitting
            callbacks=self.get_callbacks(),        # Apply anti-overfitting callbacks
            verbose=1                              # Print progress
        )

        # Best res
        best_epoch = self.history.history["val_accuracy"].index(max((self.history.history["val_accuracy"])))
        best_acc = self.history.history["val_accuracy"][best_epoch]

        print(f"Best accuracy: {best_acc} (epoch: {best_epoch})")

    def evaluate(self):
        # Evaluate the model on the test dataset
        test_loss, test_accuracy = self.conv.model.evaluate(self.test_data)

        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}\n")

    def plot_history(self):
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 4))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def save_model(self):
        # First, save as a .h5
        self.conv.model.save("classifier_final.h5")
        print("\nFinal model saved as 'final_school_classifier.h5'")

if __name__ == "__main__":
    trainer = Trainer()

    trainer.train()
    trainer.evaluate()

    trainer.plot_history()
    trainer.save_model()
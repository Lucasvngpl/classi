import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from model import Conv
from load import load_all

class Trainer:
    def __init__(self):
    

        self.conv = Conv()
        self.conv.compile_model()

        # Placeholders for the three datasets
        self.train_data = train_ds      # Training set (70%)
        self.valid_data = valid_ds      # Validation set (15%) - used to detect overfitting
        self.test_data = test_ds       # Test set (15%) - final unseen evaluation

        # place holder for load function
    
    def get_callbacks(self):
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",      # Monitor validation loss
            patience=3,              # Stop after 3 epochs of no improvement
            restore_best_weights=True # Restore model weights from the epoch with the best validation loss
        )

        return [early_stopping]

    def train(self):
        # check if loaded
        if train_ds is None or valid_ds is None or test_ds is None:
            raise ValueError("Datasets not loaded. Please load datasets before training.")
        
        # Train the model
        # fit() updates model weights using TRAINING data
        # validation_data is used ONLY for monitoring (no weight updates)
        self.history = self.conv.model.fit( # Saves the training results returned by fit() into self.history.
            self.train_data,                       # Training data - model learns from this
            epochs=self.EPOCHS,                    # HYPERPARAMETER: Number of full passes through data
            validation_data=self.valid_data,       # Validation data - detects overfitting
            callbacks=self.get_callbacks(),        # Apply anti-overfitting callbacks
            verbose=1                              # Print progress
        )

        # After training, we can check if overfitting occurred by comparing
        # training accuracy vs validation accuracy in the history
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50 + "\n")
    
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
        """Save the final trained model to disk"""
        self.conv.model.save('final_school_image_classifier.h5')
        print("\nFinal model saved as 'final_school_classifier.h5'")

if __name__ == "__main__":
    trainer = Trainer()
    # trainer.load_data()  # Implement data loading method
    trainer.train()
    trainer.evaluate()
    trainer.plot_history()
    trainer.save_model()


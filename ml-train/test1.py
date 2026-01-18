import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from model_params import *


class ModelTester:
    def __init__(self, model_path="classifier_final.h5"):
        """Load the trained model"""
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully!\n")

        # Load test data
        print("Loading test data...")
        self.test_ds = self.load_test_data()
        self.class_names = ['academic', 'non_academic']

    def load_test_data(self, fp="../dataset-full/test"):
        """Load test dataset"""
        test_ds = keras.utils.image_dataset_from_directory(
            fp,
            labels="inferred",
            label_mode="binary",
            batch_size=1,  # Load one at a time for analysis
            image_size=(IMG_SIZE, IMG_SIZE),
            shuffle=False  # Keep order consistent
        )

        # Normalize
        norm = tf.keras.layers.Rescaling(1.0 / 255)
        test_ds = test_ds.map(lambda x, y: (norm(x), y))

        return test_ds

    def evaluate_all(self):
        """Evaluate model on all test images and collect results"""
        print("Evaluating model on test set...\n")

        results = {
            'images': [],
            'true_labels': [],
            'predictions': [],
            'confidences': [],
            'correct': []
        }

        for images, labels in self.test_ds:
            # Get prediction
            pred = self.model.predict(images, verbose=0)[0][0]
            true_label = int(labels.numpy()[0])
            pred_label = 1 if pred > 0.5 else 0

            # Store results
            results['images'].append(images[0].numpy())
            results['true_labels'].append(true_label)
            results['predictions'].append(pred_label)
            results['confidences'].append(pred)
            results['correct'].append(pred_label == true_label)

        self.results = results
        return results

    def print_summary(self):
        """Print summary statistics"""
        correct = sum(self.results['correct'])
        total = len(self.results['correct'])
        accuracy = correct / total * 100

        print("=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total images: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Incorrect predictions: {total - correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print()

        # Per-class accuracy
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = [i for i, label in enumerate(self.results['true_labels'])
                          if label == class_idx]
            class_correct = sum([self.results['correct'][i] for i in class_mask])
            class_total = len(class_mask)
            class_acc = class_correct / class_total * 100 if class_total > 0 else 0

            print(f"{class_name.upper()}:")
            print(f"  Total: {class_total}")
            print(f"  Correct: {class_correct}")
            print(f"  Accuracy: {class_acc:.2f}%")

        print("=" * 60)

    def show_incorrect_predictions(self, max_display=20):
        """Display images that were predicted incorrectly"""
        incorrect_indices = [i for i, correct in enumerate(self.results['correct'])
                             if not correct]

        if len(incorrect_indices) == 0:
            print("\nNo incorrect predictions! Perfect score!")
            return

        print(f"\n{len(incorrect_indices)} incorrect predictions found.")
        print(f"Displaying up to {max_display} examples...\n")

        n_display = min(len(incorrect_indices), max_display)
        cols = 4
        rows = (n_display + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if n_display > 1 else [axes]

        for idx, i in enumerate(incorrect_indices[:max_display]):
            img = self.results['images'][i]
            true_label = self.results['true_labels'][i]
            pred_label = self.results['predictions'][i]
            confidence = self.results['confidences'][i]

            axes[idx].imshow(img)
            axes[idx].axis('off')

            true_name = self.class_names[true_label]
            pred_name = self.class_names[pred_label]

            title = f"True: {true_name}\nPred: {pred_name}\nConf: {confidence:.2f}"
            axes[idx].set_title(title, color='red', fontsize=10)

        # Hide unused subplots
        for idx in range(n_display, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('incorrect_predictions.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to 'incorrect_predictions.png'")
        plt.show()

    def show_correct_predictions(self, max_display=20):
        """Display images that were predicted correctly"""
        correct_indices = [i for i, correct in enumerate(self.results['correct'])
                           if correct]

        if len(correct_indices) == 0:
            print("\nNo correct predictions found.")
            return

        print(f"\n{len(correct_indices)} correct predictions found.")
        print(f"Displaying up to {max_display} examples...\n")

        n_display = min(len(correct_indices), max_display)
        cols = 4
        rows = (n_display + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if n_display > 1 else [axes]

        for idx, i in enumerate(correct_indices[:max_display]):
            img = self.results['images'][i]
            true_label = self.results['true_labels'][i]
            confidence = self.results['confidences'][i]

            axes[idx].imshow(img)
            axes[idx].axis('off')

            true_name = self.class_names[true_label]

            title = f"Label: {true_name}\nConf: {confidence:.2f}"
            axes[idx].set_title(title, color='green', fontsize=10)

        # Hide unused subplots
        for idx in range(n_display, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('correct_predictions.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to 'correct_predictions.png'")
        plt.show()

    def show_low_confidence_predictions(self, threshold=0.6, max_display=20):
        """Show predictions where model is uncertain (confidence near 0.5)"""
        uncertain_indices = [i for i, conf in enumerate(self.results['confidences'])
                             if abs(conf - 0.5) < (0.5 - threshold)]

        if len(uncertain_indices) == 0:
            print(f"\nNo uncertain predictions (all confidences > {threshold}).")
            return

        print(f"\n{len(uncertain_indices)} uncertain predictions found (confidence < {threshold}).")
        print(f"Displaying up to {max_display} examples...\n")

        n_display = min(len(uncertain_indices), max_display)
        cols = 4
        rows = (n_display + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if n_display > 1 else [axes]

        for idx, i in enumerate(uncertain_indices[:max_display]):
            img = self.results['images'][i]
            true_label = self.results['true_labels'][i]
            pred_label = self.results['predictions'][i]
            confidence = self.results['confidences'][i]
            correct = self.results['correct'][i]

            axes[idx].imshow(img)
            axes[idx].axis('off')

            true_name = self.class_names[true_label]
            pred_name = self.class_names[pred_label]

            color = 'green' if correct else 'red'
            title = f"True: {true_name}\nPred: {pred_name}\nConf: {confidence:.2f}"
            axes[idx].set_title(title, color=color, fontsize=10)

        # Hide unused subplots
        for idx in range(n_display, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('uncertain_predictions.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to 'uncertain_predictions.png'")
        plt.show()

    def export_results_to_csv(self, filename='test_results.csv'):
        """Export all results to CSV for further analysis"""
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Index', 'True_Label', 'Predicted_Label',
                             'Confidence', 'Correct'])

            for i in range(len(self.results['correct'])):
                true_name = self.class_names[self.results['true_labels'][i]]
                pred_name = self.class_names[self.results['predictions'][i]]

                writer.writerow([
                    i,
                    true_name,
                    pred_name,
                    f"{self.results['confidences'][i]:.4f}",
                    self.results['correct'][i]
                ])

        print(f"Results exported to '{filename}'")

    def analyze_confidence_distribution(self):
        """Show distribution of prediction confidences"""
        confidences = self.results['confidences']
        correct = self.results['correct']

        correct_conf = [conf for i, conf in enumerate(confidences) if correct[i]]
        incorrect_conf = [conf for i, conf in enumerate(confidences) if not correct[i]]

        plt.figure(figsize=(12, 5))

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Prediction Confidence Distribution')
        plt.legend()
        plt.axvline(x=0.5, color='black', linestyle='--', label='Decision boundary')

        # Box plot
        plt.subplot(1, 2, 2)
        data = [correct_conf, incorrect_conf]
        plt.boxplot(data, labels=['Correct', 'Incorrect'])
        plt.ylabel('Confidence')
        plt.title('Confidence by Correctness')
        plt.axhline(y=0.5, color='black', linestyle='--')

        plt.tight_layout()
        plt.savefig('confidence_distribution.png', dpi=150, bbox_inches='tight')
        print("Saved confidence analysis to 'confidence_distribution.png'")
        plt.show()


if __name__ == "__main__":
    # Initialize tester
    tester = ModelTester("../models/final/classifier_final.h5")

    # Run evaluation
    tester.evaluate_all()

    # Print summary
    tester.print_summary()

    # Show visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    tester.show_correct_predictions(max_display=20)
    tester.show_incorrect_predictions(max_display=20)
    tester.show_low_confidence_predictions(threshold=0.6, max_display=20)
    tester.analyze_confidence_distribution()

    # Export to CSV
    tester.export_results_to_csv()

    print("\nDone! Check the generated PNG files and CSV.")
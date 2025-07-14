# visualizer.py

import matplotlib.pyplot as plt
import os

def plot_and_save_metrics(history, output_dir):
    """
    Visualizes training and validation metrics and saves the plot to a file.

    Args:
        history (list of dict): A list where each dictionary contains metrics for one epoch.
                                Expected keys: 'epoch', 'train_loss', 'val_accuracy', 'val_f1'.
        output_dir (str): The directory where the plot image will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract metrics from the history list
    epochs = [item['epoch'] for item in history]
    train_losses = [item['train_loss'] for item in history]
    val_accuracies = [item['val_accuracy'] for item in history]
    val_f1s = [item['val_f1'] for item in history]

    # Create a figure with two subplots, sharing the x-axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Plot Training Loss on the first y-axis (left) ---
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, train_losses, 'o-', color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    # --- Plot Validation Metrics on the second y-axis (right) ---
    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Accuracy / F1', color=color)
    ax2.plot(epochs, val_accuracies, 's--', color=color, label='Validation Accuracy')
    ax2.plot(epochs, val_f1s, '^-.' , color='tab:green', label='Validation F1-Score')
    ax2.tick_params(axis='y', labelcolor=color)
    # Set y-axis limits for validation metrics for better readability, e.g., between 0.5 and 1.0
    ax2.set_ylim(min(val_accuracies + val_f1s) - 0.05, 1.0)

    # --- Final Touches ---
    # Add a title
    fig.suptitle('Training and Validation Metrics Over Epochs', fontsize=16)
    # Add legends
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    # Adjust layout to prevent labels from overlapping
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    plot_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(plot_path)
    print(f"Metrics plot saved to {plot_path}")
    
    # Close the plot to free up memory
    plt.close()
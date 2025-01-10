import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from datetime import datetime
import pickle


def evaluate_performance(train_losses, val_losses, all_labels, all_preds, output_dir="results/performance_evaluation"):
    """
    Evaluate the performance of the model by generating loss curves, confusion matrix, classification report, 
    precision, recall, F1-score, and plotting predictions vs true labels.

    Args:
    - train_losses: List of training losses for each epoch.
    - val_losses: List of validation losses for each epoch.
    - all_labels: True labels for the validation set.
    - all_preds: Predicted labels for the validation set.
    - output_dir: Directory to save the evaluation results.
    """
    # Create a directory for saving the results with a timestamp
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(output_dir, f"performance_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    epochs_completed = len(train_losses)
    
    # Plot training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs_completed), train_losses, label='Training Loss', color='blue')
    plt.plot(range(epochs_completed), val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_plot_path = os.path.join(results_dir, "training_validation_loss.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Training and validation loss plot saved to {loss_plot_path}")
    
    # Compute and display the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Pop", "Metal", "Rock", "Hip-Hop"],
                yticklabels=["Pop", "Metal", "Rock", "Hip-Hop"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    conf_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"Confusion matrix plot saved to {conf_matrix_path}")

    # Generate and display the classification report
    class_report = classification_report(all_labels, all_preds, target_names=["Pop", "Metal", "Rock", "Hip-Hop"])
    print("Classification Report:")
    print(class_report)

    # Save the classification report to a text file
    class_report_path = os.path.join(results_dir, "classification_report.txt")
    with open(class_report_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(class_report)
    print(f"Classification report saved to {class_report_path}")

    # Calculate and display Precision, Recall, and F1-Score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')

    # Save the evaluation metrics to a text file
    metrics_path = os.path.join(results_dir, "metrics_summary.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
    print(f"Metrics summary saved to {metrics_path}")

    # Plot Predictions vs True Labels
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(all_preds)), all_preds, color='blue', label='Predictions', alpha=0.6)
    plt.scatter(range(len(all_labels)), all_labels, color='red', label='True Labels', alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel('Genre')
    plt.title('Predictions vs True Labels')
    plt.legend(loc='upper right')
    pred_vs_true_path = os.path.join(results_dir, "predictions_vs_true_labels.png")
    plt.savefig(pred_vs_true_path)
    plt.close()
    print(f"Predictions vs true labels plot saved to {pred_vs_true_path}")

    print(f"All performance evaluation results saved in {results_dir}")


def load_results(losses_path, predictions_path):
    """
    Load the training and validation losses, as well as the predictions and true labels from pickle files.

    Args:
    - losses_path: Path to the pickle file containing training and validation losses.
    - predictions_path: Path to the pickle file containing predictions and true labels.

    Returns:
    - train_losses: List of training losses for each epoch.
    - val_losses: List of validation losses for each epoch.
    - all_preds: Predicted labels for the validation set.
    - all_labels: True labels for the validation set.
    """
    with open(losses_path, 'rb') as f:
        losses = pickle.load(f)

    with open(predictions_path, 'rb') as f:
        predictions = pickle.load(f)

    train_losses = losses['train_losses']
    val_losses = losses['val_losses']
    all_preds = predictions['all_preds']
    all_labels = predictions['all_labels']

    return train_losses, val_losses, all_preds, all_labels


if __name__ == "__main__":

    # Define paths to the saved losses and predictions
    losses_path = r'results\training\losses\losses_20241224_121202\losses.pkl'
    predictions_path = r'results\training\predictions\predictions_20241224_121202\predictions.pkl'

    # Load the losses and predictions
    train_losses, val_losses, all_preds, all_labels = load_results(losses_path, predictions_path)
    print("Train Losses:", train_losses)
    print("Validation Losses:", val_losses)
    
    # Evaluate the performance of the model
    evaluate_performance(
        train_losses=train_losses,
        val_losses=val_losses,
        all_labels=all_labels,
        all_preds=all_preds
    )
    print("Evaluation completed.")


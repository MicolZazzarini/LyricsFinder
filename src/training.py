import torch
import os
import pickle
from datetime import datetime


def train_model(model, train_loader, test_loader, optimizer, criterion, epochs=2):
    """
    Train the model using the provided data loaders, optimizer, and criterion.

    Args:
    - model: The model to be trained.
    - train_loader: DataLoader for training data.
    - test_loader: DataLoader for validation/test data.
    - optimizer: The optimizer used for model training.
    - criterion: The loss function.
    - epochs: The number of training epochs.

    Returns:
    - train_losses: A list of training losses for each epoch.
    - val_losses: A list of validation losses for each epoch.
    - all_preds: The predictions made on the validation set.
    - all_labels: The true labels of the validation set.
    """
    # Early stopping parameters
    patience = 2 # Number of epochs without improvement before stopping
    best_val_loss = float('inf') # Best validation loss observed so far
    epochs_without_improvement = 0 # Counter for epochs without improvement

    # Determine the device to run the model on (CUDA if available, otherwise CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        # Training loop
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
            logits = outputs.logits
            
            # Calculate the loss
            loss = criterion(logits, labels)
            loss.backward() # Backpropagate the gradients

            optimizer.step() # Update the model's parameters

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        print(f'Epoch {epoch + 1} - Training Loss: {epoch_train_loss:.4f}')

        # Validation loop
        model.eval()
        epoch_val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad(): # Disable gradient calculation during validation
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Calculate the loss
                loss = criterion(logits, labels)
                epoch_val_loss += loss.item()

                # Get predictions
                _, preds = torch.max(logits, dim=1)

                # Store true labels and predictions
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Average validation loss for the epoch
        epoch_val_loss /= len(test_loader)
        val_losses.append(epoch_val_loss)
        print(f'Epoch {epoch + 1} - Validation Loss: {epoch_val_loss:.4f}')

        # Early stopping logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss  # Update best validation loss
            epochs_without_improvement = 0  # Reset the counter if we find an improvement
        else:
            epochs_without_improvement += 1  # Increment the counter if no improvement

        # If no improvement for 'patience' epochs, stop early
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs without improvement.')
            break

    # Save results
    save_results(train_losses, val_losses, all_preds, all_labels)

    return train_losses, val_losses, all_preds, all_labels


def save_results(train_losses, val_losses, all_preds, all_labels, output_dir="results/training"):
    """
    Save the training and validation losses, as well as the predictions and labels.

    Args:
    - train_losses: The training losses for each epoch.
    - val_losses: The validation losses for each epoch.
    - all_preds: The predictions made on the validation set.
    - all_labels: The true labels of the validation set.
    - output_dir: The directory to save the results.

    """
    # Create the output directory 
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create the losses directory
    losses_dir = os.path.join(output_dir, f"losses")
    os.makedirs(losses_dir, exist_ok=True)
    losses_dir_1 = os.path.join(losses_dir, f"losses_{timestamp}")
    os.makedirs(losses_dir_1, exist_ok=True)
    
    # Save the losses as a pickle file
    with open(os.path.join(losses_dir_1, 'losses.pkl'), 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

    # Create the predictions directory
    predictions_dir = os.path.join(output_dir, f"predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    predictions_dir_1 = os.path.join(predictions_dir, f"predictions_{timestamp}")
    os.makedirs(predictions_dir_1, exist_ok=True)
    
    # Save the predictions and labels as a pickle file
    with open(os.path.join(predictions_dir_1, 'predictions.pkl'), 'wb') as f:
        pickle.dump({'all_preds': all_preds, 'all_labels': all_labels}, f)

    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    from modeling import run_modeling_phase
    from preprocessing import LyricsDataset
    
    # initializing model, optimizer and focal loss
    model, optimizer, criterion = run_modeling_phase()

    # Load the data loaders
    train_loader = torch.load(r'results\preprocessing\data_loaders\dataloaders_20241224_112905\train_loader.pth')
    test_loader = torch.load(r'results\preprocessing\data_loaders\dataloaders_20241224_112905\test_loader.pth')

    # Train the model
    train_losses, val_losses, all_preds, all_labels = train_model(model, train_loader, test_loader, optimizer, criterion)
    

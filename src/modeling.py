import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

# Global Variables for Singleton
_model, _optimizer, _criterion = None, None, None

def initialize_model():
    """
    Initializes the BERT model for sequence classification and its optimizer.

    Returns:
        model (nn.Module): The initialized model.
        optimizer (torch.optim.Optimizer): The optimizer for training.
    """
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=4,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        ignore_mismatched_sizes=True
    )

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        config=config,
        ignore_mismatched_sizes=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    return model, optimizer



def initialize_criterion(alpha=1, gamma=2, reduction='mean'):
    """
    Initializes the Focal Loss with the given parameters.

    Args:
        alpha (float): Weighting factor for the loss.
        gamma (float): Focusing parameter to adjust the rate of down-weighting easy examples.
        reduction (str): Specifies how to reduce the loss ('mean', 'sum', or 'none').

    Returns:
        criterion (FocalLoss): The initialized focal loss criterion.
    """
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)  # Predicted probability for the correct target
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss

    return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)



def run_modeling_phase():
    """
    Singleton accessor for the model, optimizer, and criterion.
    Ensures they are initialized only once and permits to execute the entire modeling phase.
    Returns:
        _model (nn.Module): The initialized model.
        _optimizer (torch.optim.Optimizer): The optimizer for training.
        _criterion (nn.Module): The loss criterion.
    """
    global _model, _optimizer, _criterion
    
    print("Defining model, optimizer, criterion...")
    if _model is None or _optimizer is None or _criterion is None:
        _model, _optimizer = initialize_model()
        _criterion = initialize_criterion(alpha=1, gamma=2, reduction='mean')
    
    print("Model, optimizer, and criterion initialized.")
    return _model, _optimizer, _criterion



if __name__ == "__main__":
    model, optimizer, criterion = run_modeling_phase()

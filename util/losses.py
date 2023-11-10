import torch.nn as nn 
from timm.loss import LabelSmoothingCrossEntropy
class CustomLoss(nn.Module):
    def __init__(self, smoothing, r=10):
        super().__init__()
        self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.r = r

    def forward(self, preds, targets):
        # preds shape: [batch_size, patches+1, num_class]
        # targets shape: [batch_size, num_patches+1]

        batch_size, num_patches_plus_one, num_class = preds.shape

        # Separate class label predictions and position predictions
        class_preds = preds[:, 0, :]  # [batch_size, num_class] for class labels
        pos_preds = preds[:, 1:, :]  # [batch_size, num_patches, num_class] for positions

        # Separate class labels and position labels
        class_targets = targets[:, 0].long()  # [batch_size] for class labels
        pos_targets = targets[:, 1:].reshape(-1).long()  # [batch_size * num_patches] for positions

        # Calculate loss for class labels and positions separately
        class_loss = self.criterion(class_preds, class_targets)
        pos_loss = self.criterion(pos_preds.reshape(-1, num_class), pos_targets)

        # Combine the losses with the given weighting
        final_loss =  class_loss + self.r * pos_loss

        return final_loss

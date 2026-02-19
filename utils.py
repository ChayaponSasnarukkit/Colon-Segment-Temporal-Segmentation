import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        """
        Args:
            weight (Tensor, optional): A manual rescaling weight given to each class. 
                                       Must be a Tensor of size C (number of classes).
            gamma (float): The focusing parameter. Higher values strictly penalize easy examples.
            ignore_index (int): Target value that is ignored and does not contribute to the loss.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs shape:  [Batch, Num_Classes, Chunk_Size] (or 2D [Batch, Num_Classes])
        # targets shape: [Batch, Chunk_Size]              (or 1D [Batch])
        
        # 1. Compute log probabilities
        log_p = F.log_softmax(inputs, dim=1)
        
        # 2. Get standard Cross-Entropy Loss (incorporating alpha/class weights if provided)
        # We use reduction='none' so we can apply the focal modulation element-wise
        ce_loss = F.nll_loss(
            log_p, 
            targets, 
            weight=self.weight, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )
        
        # 3. Extract p_t (the probability the model assigned to the true class)
        # We compute unweighted NLL to mathematically extract pure log(p_t) without weight distortions
        unweighted_ce = F.nll_loss(
            log_p, 
            targets, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )
        p_t = torch.exp(-unweighted_ce)
        
        # 4. Apply the Focal Loss formula: (1 - p_t)^gamma * ce_loss
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        # 5. Apply the requested reduction (masking out the ignore_index)
        if self.reduction == 'mean':
            mask = targets != self.ignore_index
            # Only calculate the mean over valid, non-ignored elements
            return focal_loss[mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=inputs.device)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

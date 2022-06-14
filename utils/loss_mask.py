import torch

def mse_mask_loss(logits , target):
    logits_mask = torch.clone(logits)
    target_mask = torch.clone(target)
    
    logits_mask[torch.isnan(target)] = 0
    target_mask[torch.isnan(target)] = 0
    difference = torch.square(logits_mask - target_mask)
    
    mse_loss = torch.sum(difference) 
    missing_total = torch.sum(~torch.isnan(target))
    return mse_loss, missing_total
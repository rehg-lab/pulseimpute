import torch
import torch.nn.functional as F



def l2_mpc_loss(logits , target, residuals=False):
    logits_mpc = torch.clone(logits)
    target_mpc = torch.clone(target)
    logits_mpc[torch.isnan(target)] = 0
    target_mpc[torch.isnan(target)] = 0
    difference = torch.square(logits_mpc - target_mpc)
    l2_loss = torch.sum(difference)
    
    if residuals:
        return l2_loss, difference[~torch.isnan(target)]
    else:
        return l2_loss


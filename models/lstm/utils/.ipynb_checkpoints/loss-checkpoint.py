import torch
import torch.nn.functional as F

def berhu_mpc_loss(logits , target, threshold=1):
    logits_mpc = torch.clone(logits)
    target_mpc = torch.clone(target)
    
    logits_mpc[torch.isnan(target)] = 0
    target_mpc[torch.isnan(target)] = 0
    difference = torch.abs(logits_mpc - target_mpc)

    
    l1 = F.threshold(difference, threshold, 0)
    l2 = -F.threshold(-difference, -threshold, 0)
    l2 = (torch.square(l2)+threshold**2)/(2*threshold)

    berhu_loss = l1 + l2
    berhu_loss = torch.sum(berhu_loss)
        
    return berhu_loss

def l1_mpc_loss(logits , target, threshold=1):
    logits_mpc = torch.clone(logits)
    target_mpc = torch.clone(target)
    
    logits_mpc[torch.isnan(target)] = 0
    target_mpc[torch.isnan(target)] = 0
    difference = torch.abs(logits_mpc - target_mpc)
    
    l1_loss = torch.sum(difference)
        
    return l1_loss


def l2_mpc_loss(logits , target, threshold=1):
    logits_mpc = torch.clone(logits)
    target_mpc = torch.clone(target)
    
    logits_mpc[torch.isnan(target)] = 0
    target_mpc[torch.isnan(target)] = 0
    difference = torch.square(logits_mpc - target_mpc)
    
    l1_loss = torch.sum(difference)
        
    return l1_loss
import torch
import torch.nn.functional as F

def class_total_correct(logits, labels):
    # logits tensor [batch_size, total_classes]
    # labels, integers from 0 to total_classes-1 tensor [batch size] 
    return torch.sum(torch.argmax(logits, dim=1) == labels)

def combine_ssp_mpcsoftmax_loss(mpc_logits, target, logits, ss_labels):
    mpcsoftmax_loss = softmax_mpc_loss(mpc_logits, target)
    ssp_loss = same_sequence_pred_loss(logits, ss_labels)
    return mpcsoftmax_loss+ssp_loss, ssp_loss, mpcsoftmax_loss

def combine_ssp_chanpred_mpcsoftmax_loss(mpc_logits, target, logits, ss_labels, chanpred_logits, true_channels):
    mpcsoftmax_loss = softmax_mpc_loss(mpc_logits, target)
    chanpred_loss = channel_pred_loss(chanpred_logits, true_channels)
    ssp_loss = same_sequence_pred_loss(logits, ss_labels)
    return mpcsoftmax_loss+ssp_loss+chanpred_loss, ssp_loss, chanpred_loss, mpcsoftmax_loss

def softmax_mpc_loss(logits , target):
    logits_mpc = torch.clone(logits).transpose(1,2)
    target_mpc = torch.clone(target).squeeze(-1)
    target_mpc[torch.isnan(target_mpc)] = -100
    target_mpc = target_mpc.long()
    return F.cross_entropy(logits_mpc, target_mpc, ignore_index=-100)

def combine_ssp_chanpred_mpcl2_loss(mpc_logits, target, logits, ss_labels, chanpred_logits, true_channels):
    mpcl2_loss = l2_mpc_loss(mpc_logits, target) 
    chanpred_loss = channel_pred_loss(chanpred_logits, true_channels)
    ssp_loss = same_sequence_pred_loss(logits, ss_labels)
    return mpcl2_loss+ssp_loss+chanpred_loss, ssp_loss, chanpred_loss, mpcl2_loss

def combine_ssp_chanpred_mpcl1_loss(mpc_logits, target, logits, ss_labels, chanpred_logits, true_channels):
    mpcl1_loss = l1_mpc_loss(mpc_logits, target) 
    chanpred_loss = channel_pred_loss(chanpred_logits, true_channels)
    ssp_loss = same_sequence_pred_loss(logits, ss_labels)
    return mpcl1_loss+ssp_loss+chanpred_loss, ssp_loss, chanpred_loss, mpcl1_loss

def combine_chanpred_mpcl1_loss(mpc_logits, target, chanpred_logits, true_channels):
    mpcl1_loss = l1_mpc_loss(mpc_logits, target) 
    chanpred_loss = channel_pred_loss(chanpred_logits, true_channels)
    return mpcl1_loss+chanpred_loss, mpcl1_loss, chanpred_loss

def combine_ssp_mpcl1_loss(mpc_logits, target, logits, ss_labels):
    mpcl1_loss = l1_mpc_loss(mpc_logits, target) 
    ssp_loss = same_sequence_pred_loss(logits, ss_labels)
    return mpcl1_loss+ssp_loss, mpcl1_loss, ssp_loss

def same_sequence_pred_loss(logits, ss_labels):
    return F.cross_entropy(logits, ss_labels)

def channel_pred_loss(logits, true_channels):
    return F.cross_entropy(logits, true_channels)
    
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

def l1_mpc_loss(logits, target):
    logits_mpc = torch.clone(logits)
    target_mpc = torch.clone(target)
    
    logits_mpc[torch.isnan(target)] = 0
    target_mpc[torch.isnan(target)] = 0
    difference = torch.abs(logits_mpc - target_mpc)
    
    l1_loss = torch.sum(difference)
        
    return l1_loss


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


def l4_mpc_loss(logits , target):
    logits_mpc = torch.clone(logits)
    target_mpc = torch.clone(target)
    
    logits_mpc[torch.isnan(target)] = 0
    target_mpc[torch.isnan(target)] = 0
    difference = torch.pow(logits_mpc - target_mpc, exponent=4)
    
    l1_loss = torch.sum(difference)
        
    return l1_loss

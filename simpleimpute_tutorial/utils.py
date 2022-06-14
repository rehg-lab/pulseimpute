from inits import * 
import tqdm
import torch.nn.functional as F
from scipy.linalg import fractional_matrix_power
import torch 

class ECGdataset(torch.utils.data.Dataset):
    def __init__(self,waveform,label):
        self.waveform = waveform
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        X = self.waveform[idx,:,:]
        y = self.label[idx]
        return X,y


def train(device,model,optimizer,criterion,batch_size,learning_rate,epoch,epochs,trainloader,numsamples):
    """
    - method to train MLP for intervention prediction 
    - this function iterates over all batches (one epoch)
    """
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch,[data,_] in enumerate(trainloader):
        data = data.to(device)
        data = data.float()
        optimizer.zero_grad()
        out_pred = model(data)
        loss = criterion(out_pred,data) #cross entropy loss

        loss.backward() 
        optimizer.step()
        total_loss += loss.item()
        log_interval = 200

        if batch % log_interval == 0 and batch > 0:
            curr_loss = total_loss/log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} /{:3d}| {:5d}/{:5d} batches | lr {:02.2f}]ms/batch {:5.2f} | loss {:5.2f} | '.format(epoch,epochs,batch,numsamples//batch_size,learning_rate,elapsed*1000/log_interval,curr_loss))
            total_loss = 0
            start_time = time.time()

    return total_loss


def evaluate(device,eval_model,criterion,bs,lr,epoch,epochs,valloader,numsamples):
    """
    - method to evaluate a trained model for intervention prediction
    - this function runs over all batches (one epoch)
    - computes the accuracy, auc and confusion matrix 
    """
    eval_model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch, [data,_] in enumerate(valloader):
            data = data.to(device)
            data = data.float()
            
            out_pred = eval_model(data)

            total_loss += criterion(out_pred,data).item()

    return total_loss/(numsamples-1)



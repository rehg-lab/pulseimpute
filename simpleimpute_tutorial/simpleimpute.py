from inits import *
from utils import train,evaluate,ECGdataset
from model import lstmModel


def main(args):
    #load data here 
    X_train = np.random.rand(1000,100,1) #bs x t x f
    y_train = np.random.rand(1000,)
    X_val = np.random.rand(200,100,1) #bs x t x f
    y_val = np.random.rand(200,)
    X_test = np.random.rand(200,100,1) #bs x t x f
    y_test = np.random.rand(200,)

    trainDataset = ECGdataset(X_train,y_train)
    valDataset = ECGdataset(X_val,y_val)
    testDataset = ECGdataset(X_test,y_test)

    trainloader = torch.utils.data.DataLoader(trainDataset,batch_size=args.bs)
    valloader = torch.utils.data.DataLoader(valDataset,batch_size=args.bs)
    testloader = torch.utils.data.DataLoader(testDataset,batch_size=args.bs)

    log_path = args.savedir + 'log/'
    model_path = args.savedir + 'models/'

    #model parameters
    num_layers = args.num_layers
    lr = args.lr 
    epochs = args.epochs
    bs = args.bs
    hidden_dim = args.hidden_dim
    inp_dim = X_train.shape[2]

    model = lstmModel(num_layers=num_layers,inp_dim=inp_dim,emb_dim1=hidden_dim)
    model_name = 'lstm'
    
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    
    criterion = nn.MSELoss(reduction='mean')
    writer = SummaryWriter(log_path+model_name)
    best_val_loss = float("inf")

    for epoch in range(1,epochs+1): 
        epoch_start_time = time.time()
        
        train_loss = train(device,model,optimizer,criterion,bs,lr,epoch,epochs,trainloader,len(trainDataset))
        val_loss = evaluate(device,model,criterion,bs,lr,epoch,epochs,valloader,len(valDataset))
                
        writer.add_scalar('Loss/train',train_loss,epoch)
        writer.add_scalar('Loss/val',val_loss,epoch)
        print('-'*95)
        print('|end of epoch {:3d}| time: {:5.2f}s| valid loss {:5.2f} |'.format(epoch,(time.time()-epoch_start_time),val_loss))
        print('-'*95)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

    test_loss = evaluate(device,best_model,criterion,bs,lr,epoch,epochs,testloader,len(testDataset))
    print('='*95)
    print('|end of training {:3d}| time: {:5.2f}s| test loss {:5.2f} |'.format(epoch,(time.time()-epoch_start_time),test_loss))
        
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    torch.save(best_model.state_dict(),model_path+model_name+'.pkl')
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline without any domain adaptation')
    
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                                    help='number of total epochs to run')
    parser.add_argument('--bs',default=32,type=int,
            help='batch size')
    parser.add_argument('--device',default=0,type=int,
            help='GPU: 0 - 7')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--num_layers', default=4, type=int,
                        help='lstm layers')
    parser.add_argument('--hidden_dim', default=50, type=int,
                        help='Hidden size in lstm')
    parser.add_argument('--savedir', default='./',type=str,
                        help='directory to save logs,model')
    args = parser.parse_args() 
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
    main(args)
    #python simpleimpute.py --epochs 20 --bs 64 --device 0 

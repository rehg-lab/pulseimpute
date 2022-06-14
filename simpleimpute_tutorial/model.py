from inits import * 

class lstmModel(nn.Module):
    def __init__(self,num_layers,inp_dim,emb_dim1):
        super(lstmModel,self).__init__()
        self.lstm = nn.LSTM(inp_dim,emb_dim1,num_layers,batch_first=True)
        self.fc = nn.Linear(emb_dim1,inp_dim)
        

    def forward(self,x):
        lstm_out, _ = self.lstm(x) #bs,t,f
        fc_out = self.fc(lstm_out)
        return fc_out



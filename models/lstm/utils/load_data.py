import os
import pandas as pd
import numpy as np
import wfdb
import ast
import torch
from torch.distributions.continuous_bernoulli import ContinuousBernoulli
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import multiprocessing
import itertools
import queue 
# path should be path to mimic_waveforms folder
def load_data_mimic(window=5, batch_size=1024,path = os.path.basename(os.getcwd()), ssp=False, 
              noise=False, chanpred=False, noV=False, sspaug_range=None, mode=False,
              num_workers=0, bounds=None, discretize=None, fast=False, seed=10, reproduce=False, npy=False):

    np.random.seed(seed)
    if not chanpred:
        ecg_names = np.array([line.strip() for line in open(os.path.join(path,"valid_ecgs_all.txt"))])
    else:
        if noV:
            ecg_names = np.array([line.strip() for line in open(os.path.join(path,"valid_ecgs_all_chanpred_noV.txt"))])
        else:
            ecg_names = np.array([line.strip() for line in open(os.path.join(path,"valid_ecgs_all_chanpred.txt"))])
    
    indices = np.arange(len(ecg_names)).astype(int)

    np.random.shuffle(indices)
    # Split data into train and test
    cutoff = len(indices)*9//10
    # Train
    indices_train = indices[:cutoff]
    ecg_names_train = ecg_names[indices_train]
    # Test
    indices_test = indices[cutoff:]
    ecg_names_test = ecg_names[indices_test]

    if reproduce:
        shuffle_train=False
    else:
        shuffle_train = True

    if sspaug_range:
        train_dataset = MIMIC_Dataset(datapath=path, waveformnames=ecg_names_train, window=window, ssp=ssp, chanpred=chanpred, 
                                  noise=noise,bounds=bounds, discretize=discretize, mode=mode,
                                  shuffle=shuffle_train, sspaug_range=sspaug_range, batch_size=batch_size, reproduce=reproduce, npy=npy)
        # turn off shuffle for dataloader, we will be shuffling via prefetching
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dataset = MIMIC_Dataset(datapath=path, waveformnames=ecg_names_test, window=window, ssp=ssp, chanpred=chanpred, 
                                  noise=noise,bounds=bounds, discretize=discretize, mode=mode,
                                  shuffle=False, sspaug_range=sspaug_range, batch_size=batch_size, reproduce=True, npy=npy)
    else:
        train_dataset = MIMIC_Dataset(datapath=path, waveformnames=ecg_names_train, window=window, ssp=ssp, chanpred=chanpred, 
                                  noise=noise,bounds=bounds, discretize=discretize, mode=mode,reproduce=reproduce, npy=npy)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        test_dataset = MIMIC_Dataset(datapath=path, waveformnames=ecg_names_test, window=window, ssp=ssp, chanpred=chanpred, 
                                  noise=noise,bounds=bounds, discretize=discretize, mode=mode,reproduce=True, npy=npy)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

                   

class MIMIC_Dataset(torch.utils.data.Dataset):
    def __init__(self, datapath, waveformnames, ssp=False, chanpred=False, noise=False, window=5, bounds=None, discretize=None, mode=False, npy=False,
                shuffle=False, sspaug_range=None, batch_size=None, reproduce=False):
        'Initialization'
        self.datapath=datapath
        self.waveformnames=waveformnames
        self.window = window
        self.ssp = ssp
        self.chanpred = chanpred
        if chanpred:
            allleads = np.array([waveformname.split("_")[-1] for waveformname in waveformnames])
            self.lead_dict = partition(allleads)
            self.chan_2_label = {
                "AVF": 0,
                "AVL": 1,
                "AVR": 2,
                "I": 3,
                "II": 4,
                "III": 5,
                "MCL": 6,
                "V": 7
            }
        self.mode = mode
        self.noise = noise
        self.epoch = -1
        self.bounds = bounds
        self.discretize = discretize
        self.endofseqidx = None
        self.npy = npy
        self.shuffle = shuffle
        self.sspaug_range = sspaug_range
        self.batch_size = batch_size
        self.reproduce = reproduce
        if discretize:
            self.discretize += 1
            assert((self.discretize) % 2 == 0)
            self.zeroval = (self.discretize)//2 - 1
        else:
            self.zeroval = 0

    def setup_batches(self):
        if self.shuffle:
            self.true_indexes = np.arange(len(self)).astype(int)
            np.random.shuffle(self.true_indexes)
        if self.sspaug_range:
            total_batches = len(self) // self.batch_size + 1
            self.endofseqidx_list = []
            for batch_num in range(total_batches):
                self.endofseqidx_list.append(np.random.randint(self.sspaug_range[0],self.sspaug_range[1]))

    def set_endofseqidx(self, endofseqidx):
        self.endofseqidx = endofseqidx

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.waveformnames)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # Load data and get label
        if self.reproduce:
            np.random.seed(idx)

        if self.sspaug_range:
            self.set_endofseqidx(self.endofseqidx_list[idx // self.batch_size])
        if self.shuffle:
            idx = self.true_indexes[idx]

        waveformname = self.waveformnames[idx]
        if self.npy:
            lead_signal = np.load(os.path.join(self.datapath, "mimic_100hz_npy", waveformname + ".npy")).astype(np.float32) 
        else:
            lead_signal, _ =  wfdb.rdsamp(os.path.join(self.datapath, "mimic_100hz", waveformname))


        # 10 seconds at 100 Hz
        rangeofpossiblestarts = len(lead_signal) - 100*10
        if rangeofpossiblestarts == 0:
            startidx = 0
        else:
            startidx = np.random.randint(rangeofpossiblestarts)
        lead_signal = lead_signal[startidx:startidx+100*10, :].astype(np.float32) # shape = [1000, 1 channel]
        
        if self.mode:
            hist, bin_edges = np.histogram(lead_signal, bins=50) # hist shape [50, ]
            mode = find_mode(hist, bin_edges)
            lead_signal -= mode
            if self.bounds:
                max_val = np.max(np.abs(lead_signal))
                lead_signal /= max_val/self.bounds
        elif self.bounds:
            lead_signal = bounds(lead_signal, self.bounds)
        else:
            lead_signal = lead_signal - np.mean(lead_signal)
        if self.discretize:
            lead_signal = discretize(lead_signal, self.bounds, self.discretize)

        if self.endofseqidx is None:
            endofseqidx = 100*5
        else:
            endofseqidx = self.endofseqidx 

        if self.ssp:
            rand = np.random.random_sample()
            if rand <= 0.5:
                ss_label = 0
                
                if self.chanpred: # make sure the waveform being subbed is the same channel
                    waveformname_sub = self.waveformnames[np.random.choice(self.lead_dict[waveformname.split("_")[-1]])]
                else:
                    waveformname_sub = self.waveformnames[np.random.randint(len(self.waveformnames))]

                while waveformname_sub.split("_")[0] == waveformname.split("_")[0]: # must be different patient
                    waveformname_sub = self.waveformnames[np.random.randint(len(self.waveformnames))]
               
                if self.npy:
                    lead_signal_sub = np.load(os.path.join(self.datapath, "mimic_100hz_npy", waveformname_sub+ ".npy")).astype(np.float32)
                else:
                    lead_signal_sub, _ =  wfdb.rdsamp(os.path.join(self.datapath, "mimic_100hz", waveformname_sub))


                # startidx_sub = np.random.randint(len(lead_signal_sub) - endofseqidx) # fixed the bounding after subsetting it
                startidx_sub = np.random.randint(len(lead_signal_sub) - lead_signal.shape[0] + endofseqidx) # fixed the bounding after subsetting it

                lead_signal_sub = lead_signal_sub[startidx_sub:startidx_sub + lead_signal.shape[0] - endofseqidx,:]

                if self.mode:
                    hist, bin_edges = np.histogram(lead_signal_sub, bins=50) # hist shape [50, ]
                    mode = find_mode(hist, bin_edges)
                    lead_signal_sub -= mode
                    if self.bounds:
                        max_val = np.max(np.abs(lead_signal_sub))
                        lead_signal_sub /= max_val/self.bounds
                elif self.bounds:
                    lead_signal_sub = bounds(lead_signal_sub, self.bounds)
                else:
                    lead_signal_sub = lead_signal_sub - np.mean(lead_signal_sub)
                if self.discretize:
                    lead_signal_sub = discretize(lead_signal_sub, self.bounds, self.discretize)

                lead_signal[endofseqidx:,:] = lead_signal_sub
                
            else:
                ss_label = 1

        X = torch.from_numpy(lead_signal)
        X_original = torch.clone(X)

        y = np.empty(X.shape, dtype=np.float32)
        y[:] = np.nan
        y = torch.from_numpy(y)
        
        # lets randomly mask!
        # iterate over channels
        for i in range(0, X.shape[1]):
            # iterate over time
            for j in range(0, X.shape[0], self.window):
                rand = np.random.random_sample()
                if rand <= 0.15:
                    if X.shape[0]-j <  self.window:
                        incr = X.shape[0]-j
                    else:
                        incr = self.window

                    y[j:j+incr, i] = X[j:j+incr, i]
                    rand = np.random.random_sample()
                    # Keep Same
                    if rand <= 0.10: 
                        continue
                    # Randomly Replace
                    elif rand <= 0.20:
                        if self.noise:
                            # 50 Hz for Powerline Interference -> 1/50 sec period -> 2 centisec per -> 2 = 2pi/B -> B = 2
                            X[j:j+incr, i] += .1*np.cos(2*np.arange(0,incr))
                        else:    
                            start_idx = np.random.randint(X.shape[0]-incr)
                            X[j:j+incr, i] = X_original[start_idx:start_idx+incr, i]
                    # Set to 0
                    else:
                        X[j:j+incr, i] = self.zeroval
        
        y_dict = {"target_seq": y}
        if self.ssp:
            y_dict["ss_label"] = ss_label
        y_dict["original"] = X_original
        if self.chanpred:
            y_dict["channel_label"] = self.chan_2_label[waveformname.split("_")[-1]]
        y_dict["endofseqidx"] = endofseqidx
        y_dict["name"] = waveformname
        y_dict["startidx"] = startidx
        
        return X, y_dict

def find_mode(hist, bin_edges):
    max_idx = np.argwhere(hist == np.max(hist))[0][0]
    mode = np.mean([bin_edges[max_idx], bin_edges[1+max_idx]])
    
    return mode

def partition(array):
    return {i: np.where(np.array(array) == i)[0] for i in np.unique(array)}

def bounds(lead_signal, bounds):
    signal_min = np.min(lead_signal)
    lead_signal -= signal_min # scaling 0 to max

    signal_max = np.max(lead_signal)
    if signal_max ==0:
        print(waveformname)

    lead_signal /= signal_max # scaling 0 to 1

    lead_signal *= bounds*2
    lead_signal -= 1 # scaling -1 to 1
    return lead_signal

def discretize(lead_signal, bounds, discretize_num):
    assert(discretize_num % 2 == 0)
    bins = np.linspace(-bounds, bounds, discretize_num)

    lead_signal = np.digitize(lead_signal, bins=bins).astype('float32')
    lead_signal[lead_signal == discretize] = discretize-1
    lead_signal -= 1 # so that nums go from 0 to discretize-1

    return lead_signal

def load_data(window=5, batch_size=1024, path = os.path.basename(os.getcwd()), single_channel=True, channel_pred=False, ssp=False, 
              ssp_samechan=False, noise=False, imputewarmup=False, 
              onechanneltrainimpute=False, onechanneltestimpute=False,
              num_workers=0, bounds=None, discretize=None, fast=False, mode=False):
    sampling_rate=100
    # load and convert annotation data
    Y = pd.read_csv(os.path.join(path,'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))


    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(os.path.join(path,'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    # clean missingness from missing aggregrate code
    Y = Y[[len(superclass) != 0 for superclass in Y.diagnostic_superclass]]

    # Load raw signal data
    print("Loading Raw Data ...")
    X = load_raw_data(Y, sampling_rate, path)
    print("Finished loading Raw Data ...")
    # Split data into train and test
    test_fold = 10
    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]

    if fast:
        X_train = X_train[:100,:,:]
        X_test = X_test[:100,:,:]

    if bounds:
        minimum_train = np.amin(X_train, axis=(1,2), keepdims=True)
        X_train -= minimum_train # scaling 0 to max
        minimum_test = np.amin(X_test, axis=(1,2), keepdims=True)
        X_test -= minimum_test # scaling 0 to max

        maximum_train = np.amax(X_train, axis=(1,2), keepdims=True)
        X_train /= maximum_train # scaling 0 to 1
        maximum_test = np.amax(X_test, axis=(1,2), keepdims=True)
        X_test /= maximum_test # scaling 0 to 1

        X_train *= bounds*2
        X_train -= 1 # scaling -1 to 1
        X_test *= bounds*2
        X_test -= 1 # scaling -1 to 1
    else:
        if mode:
            # not supported yet lmao sigh
            import sys; sys.exit()
            hist, bin_edges = np.histogram(lead_signal, bins=50) # hist shape [50, ]
            lead_mode = find_mode(hist, bin_edges)
            lead_signal -= lead_mode
        else:
            X_train = X_train - np.mean(X_train, axis = 1, keepdims=True)
            X_test = X_test - np.mean(X_test, axis = 1, keepdims=True)

    if discretize:
        assert(discretize % 2 == 1)
        discretize += 1
        bins = np.linspace(-bounds, bounds, discretize)

        X_train = np.digitize(X_train, bins=bins).astype('float32')
        X_train[X_train == discretize] = discretize-1
        X_train -= 1 # so that nums go from 0 to discretize-1

        X_test = np.digitize(X_test, bins=bins).astype('float32')
        X_test[X_test == discretize] = discretize-1
        X_test -= 1 # so that nums go from 0 to discretize-1

    if single_channel:
        X_train = np.transpose(X_train, (0,2,1))
        X_train = X_train.reshape((-1, X_train.shape[-1], 1))

        X_test = np.transpose(X_test, (0,2,1))
        X_test = X_test.reshape((-1, X_test.shape[-1], 1))

    X_train = torch.from_numpy(X_train)
    train_dataset = MPCDataset(waveforms=X_train, window=window, channel_pred=channel_pred, ssp=ssp, ssp_samechan=ssp_samechan, noise=noise,
                               onechannelimpute=onechanneltrainimpute, imputewarmup=imputewarmup, discretize=discretize)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    X_test = torch.from_numpy(X_test)
    test_dataset = MPCDataset_handcurated(waveforms=X_test,  window=window, channel_pred=channel_pred,  ssp=ssp, ssp_samechan=ssp_samechan, 
                                          noise=noise, onechannelimpute=onechanneltestimpute, discretize=discretize)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    X_test = X_test.cpu().detach().numpy()
   
    if single_channel:
        X_test = X_test.reshape((12, -1,  X_test.shape[1]), order = "F")
        X_test = np.transpose(X_test, (1, 2, 0))

    return train_loader, test_loader, X_test


class MPCDataset_handcurated(torch.utils.data.Dataset):
    def __init__(self, waveforms, window=5, channel_pred=False, ssp=False, ssp_samechan=False, noise=False, 
                 onechannelimpute=False, discretize=None):
        'Initialization'
        self.waveforms = waveforms
        self.onechannelimpute = onechannelimpute
        if channel_pred:
            self.channel_label = torch.from_numpy(np.resize(np.arange(12), len(self.waveforms)))
        else:
            self.channel_label = None
        self.ssp = ssp
        self.ssp_samechan = ssp_samechan
        self.noise = noise
        self.window = window
        self.endofseqidx = None
        if discretize:
            assert(discretize % 2 == 0)
            self.zeroval = discretize//2 - 1
        else:
            self.zeroval = 0

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.waveforms)

    def set_endofseqidx(self, endofseqidx):
        self.endofseqidx = endofseqidx

    def __getitem__(self, idx, window = 5):
        'Generates one sample of data'
        X = torch.clone(self.waveforms[idx])
        if self.ssp:
            if idx <= len(self.waveforms) // 2:
                ss_label = 0
                if self.ssp_samechan:
                    channel_label = self.channel_label[idx]
                    # will need to be changed later when data is more complex, rn just using how every 12 channel repeats
                    # maybe use hashmap to store list of indexes
                    randX = torch.clone(self.waveforms[np.random.randint(len(self.waveforms) / 12) * 12 + channel_label])
                else:
                    randX = torch.clone(self.waveforms[np.random.randint(len(self.waveforms))])
                # needs to be changed when variable inputs ..
                if self.endofseqidx is not None:
                    X[self.endofseqidx:X.shape[0],:] = randX[self.endofseqidx:X.shape[0],:] 
                else:    
                    X[X.shape[0]//2:X.shape[0],:] = randX[X.shape[0]//2:X.shape[0],:] 
            else:
                ss_label = 1
        X_original = torch.clone(X)
        y = np.empty(X.shape, dtype=np.float32)
        y[:] = np.nan
        y = torch.from_numpy(y)
        
        if self.onechannelimpute:
            # S wave increase in lead 9
            y[373:382,8] = X_original[373:382,8]
            X[373:382,8] = self.zeroval
            # R wave increase in lead 10
            y[752:757,9] = X_original[752:757,9]
            X[752:757,9] = self.zeroval
            # R wave decrease in lead 1
            y[685:690,0] = X_original[685:690,0]
            X[685:690,0] = self.zeroval
            # Q wave decrease in lead 3, ! wave decreases are hard to find
            y[440:445,2] = X_original[440:445,2]
            X[440:445,2] = self.zeroval
            # T wave decrease in lead 7
            y[640:645,6] = X_original[640:645,6]
            X[640:645,6] = self.zeroval
            # T wave increase segment in lead 6
            y[640:645,5] = X_original[640:645,5]
            X[640:645,5] = self.zeroval
            # ST segment in lead 8
            y[625:630,7] = X_original[625:630,7]
            X[625:630,7] =self.zeroval
            # random inbetween in all leads
            y[179:184,:] = X_original[179:184,:]
            X[179:184,:] = self.zeroval

            # valley Q wave found in lead 1
            y[530:535,0] = X_original[530:535,0]
            X[530:535,0] = self.zeroval
            # valley S wave found in lead 5
            y[278:283,4] = X_original[278:283,4]
            X[278:283,4] = self.zeroval
            # peak R wave found in lead 9
            y[848:853,8] = X_original[848:853,8]
            X[848:853,8] = self.zeroval
            # valley T wave found in lead 4
            y[75:80,3] = X_original[75:80,3]
            X[75:80,3] = self.zeroval
            # peak T wave found in lead 9
            y[155:160,8] = X_original[155:160,8]
            X[155:160,8] = self.zeroval
            # P wave found in lead 1
            y[915:920,0] = X_original[915:920,0]
            X[915:920,0] = self.zeroval
        else:
            if idx >= len(self.waveforms) - 12 - 1:
                y[373:382,:] = X_original[373:382,:]
                X[373:382,:] = self.zeroval
                y[752:757,:] = X_original[752:757,:]
                X[752:757,:] = self.zeroval
                y[685:690,:] = X_original[685:690,:]
                X[685:690,:] = self.zeroval
                y[440:445,:] = X_original[440:445,:]
                X[440:445,:] = self.zeroval
                y[640:645,:] = X_original[640:645,:]
                X[640:645,:] = self.zeroval
                y[640:645,:] = X_original[640:645,:]
                X[640:645,:] = self.zeroval
                y[625:630,:] = X_original[625:630,:]
                X[625:630,:] = self.zeroval
                y[179:184,:] = X_original[179:184,:]
                X[179:184,:] = self.zeroval
                y[530:535,:] = X_original[530:535,:]
                X[530:535,:] = self.zeroval
                y[278:283,:] = X_original[278:283,:]
                X[278:283,:] = self.zeroval
                y[848:853,:] = X_original[848:853,:]
                X[848:853,:] = self.zeroval
                y[75:80,:] = X_original[75:80,:]
                X[75:80,:] = self.zeroval
                y[155:160,:] = X_original[155:160,:]
                X[155:160,:] = self.zeroval
                y[915:920,:] = X_original[915:920,:]
                X[915:920,:] = self.zeroval
            else:
                # iterate over channels
                for i in range(0, X.shape[1]):
                    # iterate over time
                    for j in range(0, X.shape[0], self.window):
                        rand = np.random.random_sample()
                        if rand <= 0.15:
                            if X.shape[0]-j <  self.window:
                                incr = X.shape[0]-j
                            else:
                                incr = self.window

                            y[j:j+incr, i] = X[j:j+incr, i]
                            rand = np.random.random_sample()
                            # Keep Same
                            if rand <= 0.10: 
                                continue
                            # Randomly Replace
                            elif rand <= 0.20:
                                if self.noise:
                                    # 50 Hz for Powerline Interference -> 1/50 sec period -> 2 centisec per -> 2 = 2pi/B -> B = 2
                                    X[j:j+incr, i] += .1*np.cos(2*np.arange(0,incr))
                                else:    
                                    start_idx = np.random.randint(X.shape[0]-incr)
                                    X[j:j+incr, i] = X_original[start_idx:start_idx+incr, i]
                            # Set to 0
                            else:
                                X[j:j+incr, i] = self.zeroval
        
        if self.channel_label is None and self.ssp == False:
            return X, y
        else:
            y_dict = {"target_seq": y}
            if self.channel_label is not None: 
                channel_label = self.channel_label[idx]
                y_dict["channel_label"] = channel_label
            if self.ssp:
                y_dict["ss_label"] = ss_label

            return X, y_dict
    
class MPCDataset(torch.utils.data.Dataset):
    def __init__(self, waveforms, channel_pred=False, ssp=False, ssp_samechan=False, ssp_varlen=False,
                noise=False, onechannelimpute=False, imputewarmup=False, window=5, discretize=None):
        'Initialization'
        self.onechannelimpute=onechannelimpute
        self.imputewarmup=imputewarmup
        self.window = window
        self.waveforms = waveforms
        if channel_pred:
            self.channel_label = torch.from_numpy(np.resize(np.arange(12), len(self.waveforms)))
        else:
            self.channel_label = None
        self.ssp = ssp
        self.ssp_samechan = ssp_samechan
        self.endofseqidx = None
        self.noise = noise
        self.epoch = -1
        if discretize:
            assert(discretize % 2 == 0)
            self.zeroval = discretize//2 - 1
        else:
            self.zeroval = 0

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.waveforms)

    def set_endofseqidx(self, endofseqidx):
        self.endofseqidx = endofseqidx

    def __getitem__(self, idx):
        'Generates one sample of data'
        # Load data and get label
        X = torch.clone(self.waveforms[idx])
        if self.ssp:
            rand = np.random.random_sample()
            if rand <= 0.5:
                ss_label = 0
                if self.ssp_samechan:
                    channel_label = self.channel_label[idx]
                    # will need to be changed later when data is more complex, rn just using how every 12 channel repeats
                    # maybe use hashmap to store list of indexes
                    randX = torch.clone(self.waveforms[np.random.randint(len(self.waveforms) / 12) * 12 + channel_label])
                else:
                    randX = torch.clone(self.waveforms[np.random.randint(len(self.waveforms))])
                # needs to be changed when variable inputs ..
                if self.endofseqidx is not None:
                    X[self.endofseqidx:X.shape[0],:] = randX[self.endofseqidx:X.shape[0],:] 
                else:    
                    X[X.shape[0]//2:X.shape[0],:] = randX[X.shape[0]//2:X.shape[0],:] 
            else:
                ss_label = 1
        X_original = torch.clone(X)
        y = np.empty(X.shape, dtype=np.float32)
        y[:] = np.nan
        y = torch.from_numpy(y)
        
        # lets randomly mask!
        if self.imputewarmup:
            for j in range(0, X.shape[0], self.window):
                rand = np.random.random_sample()
                if rand <= 0.15:
                    if X.shape[0]-j <  self.window:
                        incr = X.shape[0]-j
                    else:
                        incr = self.window
                    # create continous bernoulli with maximum prob at .5
                    # warmup on learning more channels imputed
                    cont_bern_mod = ContinuousBernoulli(np.min([.01 + .01*self.epoch, .5]))
                    cont_bern_samp = cont_bern_mod.sample().item()
                    total_channels_masked = np.min([int(cont_bern_samp * X.shape[1] + 1), 12])
                    which_channels_masked = np.random.choice(list(np.arange(0,12)), total_channels_masked, replace=False)

                    y[j:j+incr, which_channels_masked] = X[j:j+incr, which_channels_masked]
                    rand = np.random.random_sample()
                    # Keep Same
                    if rand <= 0.10: 
                        continue
                    # Randomly Replace
                    elif rand <= 0.20:
                        start_idx = np.random.randint(X.shape[0]-incr)
                        X[j:j+incr, which_channels_masked] = X_original[start_idx:start_idx+incr, which_channels_masked]
                    # Set to 0
                    else:
                        X[j:j+incr, which_channels_masked] = 0
        else:
            # iterate over channels
            for i in range(0, X.shape[1]):
                # iterate over time
                if self.onechannelimpute:
                    i = np.arange(0, X.shape[1], dtype=np.int32).tolist()
                for j in range(0, X.shape[0], self.window):
                    rand = np.random.random_sample()
                    if rand <= 0.15:
                        if X.shape[0]-j <  self.window:
                            incr = X.shape[0]-j
                        else:
                            incr = self.window

                        y[j:j+incr, i] = X[j:j+incr, i]
                        rand = np.random.random_sample()
                        # Keep Same
                        if rand <= 0.10: 
                            continue
                        # Randomly Replace
                        elif rand <= 0.20:
                            if self.noise:
                                # 50 Hz for Powerline Interference -> 1/50 sec period -> 2 centisec per -> 2 = 2pi/B -> B = 2
                                X[j:j+incr, i] += .1*np.cos(2*np.arange(0,incr))
                            else:    
                                start_idx = np.random.randint(X.shape[0]-incr)
                                X[j:j+incr, i] = X_original[start_idx:start_idx+incr, i]
                        # Set to 0
                        else:
                            X[j:j+incr, i] = self.zeroval
                if self.onechannelimpute:
                    break
        
            
        if self.channel_label is None and self.ssp == False:
            return X, y
        else:
            y_dict = {"target_seq": y}
            if self.channel_label is not None: 
                channel_label = self.channel_label[idx]
                y_dict["channel_label"] = channel_label
            if self.ssp:
                y_dict["ss_label"] = ss_label

            return X, y_dict


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path,f)) for f in tqdm(df.filename_lr, leave=False)]
    else:
        data = [wfdb.rdsamp(os.path.join(path,f)) for f in tqdm(df.filename_hr, leave=False)]
    data = np.array([signal for signal, meta in data], dtype=np.float32)
    return data

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

class SSPaug_DataLoader():
    def __init__(
        self,
        dataset,
        sspaug_range,
        shuffle=False,
        batch_size=64,
        num_workers=1,
        prefetch_batches=2,
        collate_fn=default_collate,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.index = 0
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.output_queue = multiprocessing.Queue()
        self.index_queues = []
        self.workers = []
        self.worker_cycle = itertools.cycle(range(num_workers))
        self.cache = {}
        self.prefetch_index = 0
        self.shuffle = shuffle
        self.sspaug_range = sspaug_range

        self.data_indexes = np.arange(len(dataset)).astype(int)
        if self.shuffle:
            np.random.shuffle(self.data_indexes)
        self.total_batches = len(dataset) // batch_size + 1
        self.endofseqidx_list = []
        for batch_num in range(self.total_batches):
            self.endofseqidx_list.append(np.random.randint(self.sspaug_range[0],self.sspaug_range[1]))

        def worker_fn(dataset, index_queue, output_queue):
            while True:
                try:
                    index = index_queue.get(timeout=0)
                except queue.Empty:
                    continue
                if index is None:
                    break

                endofseqidx = self.endofseqidx_list[index // self.batch_size]
                data_index = self.data_indexes[index]
                dataset.set_endofseqidx(endofseqidx)
                output_queue.put((index, dataset[data_index]))

        for _ in range(num_workers):
            index_queue = multiprocessing.Queue()
            worker = multiprocessing.Process(
                target=worker_fn, args=(self.dataset, index_queue, self.output_queue)
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.index_queues.append(index_queue)

        self.prefetch()
    
    def __next__(self):
        if self.index >= len(self.dataset):
            # stop iteration once index is out of bounds
            raise StopIteration
        batch_size = min(len(self.dataset) - self.index, self.batch_size)
        waveforms = []
        label_dicts = []
        for _ in range(batch_size):
            waveform, label_dict = self.get()
            waveforms.append(waveform)
            label_dicts.append(label_dict)
        return self.collate_fn(waveforms), self.collate_fn(label_dicts)

    def prefetch(self):
        while (
            self.prefetch_index < len(self.dataset)
            and self.prefetch_index
            < self.index + 2 * self.num_workers * self.batch_size
        ):
            # if the prefetch_index hasn't reached the end of the dataset
            # and it is not 2 batches ahead, add indexes to the index queues
            self.index_queues[next(self.worker_cycle)].put(self.prefetch_index)
            self.prefetch_index += 1

    def get(self):
        self.prefetch()
        if self.index in self.cache:
            item = self.cache[self.index]
            del self.cache[self.index]
        else:
            while True:
                try:
                    (index, data) = self.output_queue.get(timeout=0)
                except queue.Empty:  # output queue empty, keep trying
                    continue
                if index == self.index:  # found our item, ready to return
                    item = data
                    break
                else:  # item isn't the one we want, cache for later
                    self.cache[index] = data

        self.index += 1
        return item

    def __iter__(self):
        self.index = 0
        self.cache = {}
        self.prefetch_index = 0
        self.prefetch()
        return self

    def __del__(self):
        try:
            # Stop each worker by passing None to its index queue
            for i, w in enumerate(self.workers):
                self.index_queues[i].put(None)
                w.join(timeout=5.0)
            for q in self.index_queues:  # close all queues
                q.cancel_join_thread() 
                q.close()
            self.output_queue.cancel_join_thread()
            self.output_queue.close()
        finally:
            for w in self.workers:
                if w.is_alive():  # manually terminate worker if all else fails
                    w.terminate()



class notparallel_SSPaug_DataLoader(torch.utils.data.Sampler):
    '''
    Wraps another custom sampler with epoch intervals 
    to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
                its size would be less than ``batch_size``
        epoch_size : Number of items in an epoch
    '''

    def __init__(self, dataset, batch_size, shuffle, sspaug_range, num_workers):

        self.shuffle = shuffle
        self.dataset = dataset
        self.batch_size = batch_size
        self.sspaug_range = sspaug_range
        self.num_workers

    def __iter__(self):
        batch = []
        label_dicts = []
        indices = np.arange(len(self.dataset)).astype(int)
        if self.shuffle:
            np.random.shuffle(indices)

        endofseqidx = np.random.randint(self.sspaug_range[0],self.sspaug_range[1])
        self.dataset.set_endofseqidx(endofseqidx)
        for i in indices:
            waveform, label_dict = self.dataset[i]

            batch.append(waveform)
            label_dicts.append(label_dict)

            if len(batch) == self.batch_size:
                yield default_collate(batch), default_collate(label_dicts)
                batch = []
                label_dicts = []
                endofseqidx = np.random.randint(self.sspaug_range[0],self.sspaug_range[1])
                self.dataset.set_endofseqidx(endofseqidx)
        if len(batch) > 0:
            yield batch, label_dicts

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size  # type: ignore
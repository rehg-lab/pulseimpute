# coding: utf-8

import os
import re
import numpy as np
import pandas as pd
# import ujson as json
import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from tqdm.contrib.concurrent import process_map  # or thread_map
import csv
from ast import literal_eval

train=None
waveforms = None
train_impute_wind_2 = None
train_impute_prob_2 = None
train_impute_extended_2 = None
train_real_2 = None
filename = None
fs = None
list_of_miss = None
masks = None
bigfile_2 = True

def create_dataloader(data=None, imputation_dict=None, type="train", annotate="", path="", annotate_test="", dataname="", createjson=True,
                    batch_size=4, num_workers=os.cpu_count(),
                    train_realppg=None,train_realecg=None, 
                    train_impute_wind=None, train_impute_prob=None, train_impute_extended=None,
                    bigfile=True,
                    prefetch_factor=2):

    # global variables used for parallellizing creation of brits files
    global bigfile_2
    bigfile_2 = bigfile
    global train_impute_wind_2
    train_impute_wind_2=train_impute_wind
    global train_impute_prob_2
    train_impute_prob_2=train_impute_prob
    global train_impute_extended_2
    train_impute_extended_2=train_impute_extended 
    global train_real_2
    train_real_2 = train_realecg or train_realppg

    # file creation
    global train
    if type=="train":
        train=True
    elif type=="val" or type=="test":
        train=False
    else:
        print("fix the type dude")
        import sys; sys.exit()

    global filename
    if type=="test":
        filename = f"{dataname}{type}{annotate_test}"
    else:
        annotate = "_" + "_".join(annotate.split("_")[2:])
        filename = f"{type}{annotate}"

    if train_realecg or train_realppg:
        if train_realppg:
            miss_tuples_path = os.path.join("data","missingness_patterns", f"missing_ppg_{type}.csv")
        elif train_realecg:
            miss_tuples_path = os.path.join("data","missingness_patterns", f"missing_ecg_{type}.csv")
        with open(miss_tuples_path, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            global list_of_miss
            list_of_miss = list(csv_reader)
    else: 
        list_of_miss = None

    if imputation_dict:
        global masks
        masks = imputation_dict["target_seq"]


    if createjson:
        print("creating file " + filename)
        print(os.path.join(path, filename))
        if (not os.path.exists(os.path.join(path, filename))):
            global waveforms
            waveforms = data
            global fs
            fs =os.path.join(path, filename)
            if not bigfile_2:
                os.mkdir(fs)
            print(data.shape[0])
            process_map(parse_id, np.arange(data.shape[0]), max_workers=os.cpu_count(),chunksize=1)
        dataset = BRITS_dataset(path=os.path.join(path, filename), bigfile=bigfile)
    else:
        dataset = BRITS_dataset_wrapper()

    print(len(dataset))
    data_loader = DataLoader(dataset = dataset, batch_size = batch_size, num_workers=num_workers, shuffle=train, 
                             pin_memory=True, collate_fn = collate_fn, prefetch_factor=prefetch_factor)
    
    return data_loader


def parse_id(idx):
    waveform = waveforms[idx].detach().numpy()
    shp = waveform.shape
    values = waveform.copy()
    if masks is None:
        if train_impute_extended_2:
            for channel in range(waveform.shape[1]):
                np.random.seed(idx)
                start_idx = np.random.randint(waveform.shape[0] - train_impute_extended_2)
                values[start_idx:start_idx+train_impute_extended_2, channel] = np.nan
        elif (train_impute_wind_2 and train_impute_prob_2): # randomly impute chunks
            # assert waveform.shape[0] == 1000
            for start_idx in range(0,  waveform.shape[0], train_impute_wind_2):
                for channel in range(waveform.shape[1]):
                    rand = np.random.random_sample()
                    if rand <= train_impute_prob_2:
                        values[start_idx:start_idx+train_impute_wind_2, channel] = np.nan
        elif train_real_2:
            bf_idx = np.random.randint(len(list_of_miss))
            miss_vector = miss_tuple_to_vector(list_of_miss[bf_idx])
            values[np.where(miss_vector == 0)] = np.nan
            # values = values * miss_vector
        else:        # randomly eliminate 10% values as the imputation ground-truth across all channels, this was original training method
            indices = np.where(~np.isnan(waveform))[0].tolist()
            indices = np.random.choice(indices, len(indices) // 10)
            values[indices] = np.nan
    else:
        # mask is nan at non impute areas
        mask = masks[idx]
        indices = torch.nonzero(~torch.isnan(mask)).numpy().T
        values[indices[0,:], indices[1,:]] = np.nan

    values_masks = ~np.isnan(values)
    waveform_masks = (~np.isnan(values)) ^ (~np.isnan(waveform)) #exact opposite of masks

    waveform = waveform.reshape(shp)
    values = values.reshape(shp)

    values_masks = values_masks.reshape(shp)
    waveform_masks = waveform_masks.reshape(shp)

    # waveform = evals, mask at where the signal is
    # value is nan at impute window
    # valuemask is where impute window is

    rec = {}

    rec["idx"] = int(idx)

    # prepare the model for both directions
    rec['forward'] = parse_rec(values, values_masks, waveform, waveform_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], values_masks[::-1], waveform[::-1], waveform_masks[::-1], dir_='backward')
    if train:
        rec['is_train'] = 1
    else:
        rec['is_train'] = 0

    if masks is not None:
        rec['target_seq'] = np.nan_to_num(mask).tolist()

    global fs
    if fs:
        if not bigfile_2:
            with open(os.path.join(fs, str(idx)+".txt"), 'a') as f:
                rec = json.dumps(rec)
                f.write(rec)
        else:
            with open(fs, 'a') as f:
                rec = json.dumps(rec)
                f.write(rec + '\n')
    else:
        return rec

def miss_tuple_to_vector(listoftuples):
    def onesorzeros_vector(miss_tuple):
        miss_tuple = literal_eval(miss_tuple)
        if miss_tuple[0] == 0:
            return np.zeros(miss_tuple[1])
        elif miss_tuple[0] == 1:
            return np.ones(miss_tuple[1])

    miss_vector = onesorzeros_vector(listoftuples[0])
    for i in range(1, len(listoftuples)):
        miss_vector = np.concatenate((miss_vector, onesorzeros_vector(listoftuples[i])))
    miss_vector = np.expand_dims(miss_vector, 1)
    return miss_vector


def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
        values = torch.FloatTensor(list(map(lambda r: r['values'], recs)))
        masks = torch.FloatTensor(list(map(lambda r: r['masks'], recs)))
        deltas = torch.FloatTensor(list(map(lambda r: r['deltas'], recs)))

        evals = torch.FloatTensor(list(map(lambda r: r['evals'], recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: r['eval_masks'], recs)))
        forwards = torch.FloatTensor(list(map(lambda r: r['forwards'], recs)))


        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    # ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'], recs)))
    if "target_seq" in recs[0]:
        ret_dict['target_seq'] = torch.FloatTensor(list(map(lambda x: x['target_seq'], recs)))

    return ret_dict
    
class BRITS_dataset(Dataset):
    def __init__(self, path=None, recs=None, bigfile=True):
        self.path = path
        self.bigfile = bigfile
        if path:
            if bigfile:
                self.content = open(path).readlines()
            else:
                self.content = path
        else:
            self.content = recs

    def __len__(self):
        if not self.bigfile and self.path:
            return len([name for name in os.listdir(self.content)])
        else:
            return len(self.content)

    def __getitem__(self, idx):
        if self.path:
            if self.bigfile:
                rec = json.loads(self.content[idx])
            else:
                rec = json.loads(open(os.path.join(self.path, str(idx)+".txt")).readlines()[0])
        else:
            rec = self.content[idx]
        return rec

def parse_data(x):
    x = x.set_index('Parameter').to_dict()['Value']

    values = []

    for attr in attributes:
        if x.has_key(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values


def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []


    for h in range(masks.shape[0]):
        if h == 0:
            deltas.append(np.ones(masks.shape[1])) # 35 attributes
        else: # delta is time gap from last observation to current time-stamp
            deltas.append(np.ones(masks.shape[1]) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)



def parse_rec(values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).values
    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

    return rec


class BRITS_dataset_wrapper(Dataset):
    def __init__(self):
        print("total missingness")
        print(len(list_of_miss))
    def __len__(self):
        'Denotes the total number of samples'
        return len(waveforms)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # Load data and get label
        
        return parse_id(idx)
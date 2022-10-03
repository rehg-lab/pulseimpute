import os
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from tqdm import tqdm
import csv
from ast import literal_eval


class tutorial():
    def __init__(self, modelname,
                 data_name="",
                 train_data=None, val_data=None,
                 # annotations are used for creating folders to save models in
                 annotate="", annotate_test="",
                 bs=64, gpus=[0, 1],
                 # this is passed during testing, to serve as ground truth
                 imputation_dict=None,
                 # this is passed during training, for configuring how missingness is simulated while training the imputation model
                 missingness_config=None,
                 # when reloading the model, which iteration to reload
                 reload_iter="latest",
                 # how many iterations until running validation and saving model
                 iter_save=1000
                 ):
        '''
        Constructs necessary attributes for tutorial class

                Parameters:
                        data_name (str): A decimal integer
                        train_data (int): Another decimal integer

                Returns:
                        binary_sum (str): Binary string of the sum of a and b
        '''


        outpath = "out/"
        # data loader setup
        self.bs = bs
        self.gpu_list = gpus
        self.iter_save = iter_save

        self.reload_iter = reload_iter

        self.data_loader_setup(train_data, val_data,
                               imputation_dict=imputation_dict,
                               missingness_config=missingness_config)

        if len(self.gpu_list) == 1:
            torch.cuda.set_device(self.gpu_list[0])

        # import model class
        model_module_class = getattr(__import__(
            f'models.tutorial.{modelname}', fromlist=[""]), "MainModel")
        self.model = nn.DataParallel(model_module_class(
            orig_dim=self.total_channels), device_ids=self.gpu_list)
        self.model.to(torch.device(f"cuda:{self.gpu_list[0]}"))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)

        self.ckpt_path = os.path.join(
            outpath, data_name+annotate_test, modelname+annotate)
        self.reload_ckpt_path = os.path.join(
            outpath, data_name, modelname+annotate)
        self.reload_model()

    def data_loader_setup(self, train_data, val_data=None, imputation_dict=None, missingness_config=None):
        num_threads_used = multiprocessing.cpu_count()

        if val_data is not None:
            train_dataset = dataset(
                waveforms=train_data, missingness_config=missingness_config, type="train")
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.bs, shuffle=True, num_workers=num_threads_used)

            val_dataset = dataset(
                waveforms=val_data, missingness_config=missingness_config, type="val")
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.bs, shuffle=False, num_workers=num_threads_used)
            temp = next(iter(self.val_loader))
        else:
            test_dataset = dataset(waveforms=train_data, imputation_dict=imputation_dict,
                                   missingness_config=None, type="test")
            self.test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.bs, shuffle=False, num_workers=num_threads_used)
            temp = next(iter(self.test_loader))

        self.total_channels = temp[0].shape[2]

    def reload_model(self):
        self.iter_list = [-1]
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(os.path.join(self.reload_ckpt_path,
                    "iter_latest"), exist_ok=True)
        os.makedirs(os.path.join(self.reload_ckpt_path,
                    "iter_best"), exist_ok=True)
        self.best_val_loss = 9999999
        if os.path.isfile(os.path.join(self.reload_ckpt_path, "iter_best", "iter_best.pkl")):
            state = torch.load(os.path.join(self.reload_ckpt_path, "iter_best",
                               "iter_best.pkl"), map_location=f"cuda:{self.gpu_list[0]}")
            best_iter = state["iter"]
            printlog(f"Identified best iter: {best_iter}")
            self.best_val_loss = state["l2valloss"].cpu()

        if os.path.isfile(os.path.join(self.reload_ckpt_path, f"iter_{self.reload_iter}", f"iter_{self.reload_iter}.pkl")):
            state = torch.load(os.path.join(self.reload_ckpt_path, f"iter_{self.reload_iter}", f"iter_{self.reload_iter}.pkl"),
                               map_location=f"cuda:{self.gpu_list[0]}")
            self.iter_list.append(state["iter"])

            printlog(f"Reloading given iter: {np.max(self.iter_list)}",
                     self.reload_ckpt_path)
            print(self.model.load_state_dict(state['state_dict'], strict=True))
            print(self.optimizer.load_state_dict(state['optimizer']))
        else:
            printlog(f"cannot reload iter {self.reload_iter}",
                     self.reload_ckpt_path)

    def testimp(self):
        """
        Function to compute and save the imputation error on the test dataset 
        """

        print(f'{datetime.now().strftime("%d/%m/%Y %H:%M")} | Start')

        with torch.no_grad():
            self.model.eval()
            total_test_l2_loss = 0
            total_missing_total = 0

            imputation_list = []
            for local_batch, local_label_dict in tqdm(self.test_loader, desc="Testing", leave=False):
                impute_out = self.model(local_batch)
                test_l2_loss, missing_total = l2_loss(impute_out.to(torch.device(f"cuda:{self.gpu_list[0]}")),
                                                      local_label_dict["target_seq"].to(torch.device(f"cuda:{self.gpu_list[0]}")))

                total_test_l2_loss += test_l2_loss.item()
                total_missing_total += missing_total

                impute_out[torch.where(torch.isnan(local_label_dict["target_seq"]))] = \
                    local_batch[torch.where(torch.isnan(
                        local_label_dict["target_seq"]))].cuda()
                imputation_list.append(impute_out.cpu().detach().numpy())
        total_test_l2_loss /= total_missing_total

        iter_check_path = os.path.join(
            self.ckpt_path, "iter_" + str(self.reload_iter))
        os.makedirs(iter_check_path, exist_ok=True)

        printlog(
            f'{datetime.now().strftime("%d/%m/%Y %H:%M")} | MSE:{total_test_l2_loss:.10f} \n',
            iter_check_path)

        imputation_all = np.concatenate(imputation_list, axis=0)
        np.save(os.path.join(iter_check_path, "imputation.npy"), imputation_all)

    def train(self):
        """
        model trained with the l2_loss
        """
        writer = SummaryWriter(log_dir=os.path.join(self.ckpt_path, "tb"))

        printlog(f'{datetime.now().strftime("%d/%m/%Y %H:%M")} | Start',
                 self.ckpt_path)

        iter_idx = np.max(self.iter_list)+1
        for epoch in range(0, 1000):
            total_train_l2_loss = 0
            total_missing_total = 0

            for local_batch, local_label_dict in tqdm(self.train_loader, desc="Training", leave=False):
                iter_idx += 1
                self.optimizer.zero_grad()

                impute_out = self.model(local_batch)
                train_l2_loss, missing_total = l2_loss(impute_out.to(torch.device(f"cuda:{self.gpu_list[0]}")),
                                                       local_label_dict["target_seq"].to(torch.device(f"cuda:{self.gpu_list[0]}")))

                train_l2_loss.backward()

                self.optimizer.step()
                total_train_l2_loss += train_l2_loss.item()
                total_missing_total += missing_total

                if iter_idx % self.iter_save == 0:

                    total_train_l2_loss /= total_missing_total
                    writer.add_scalar(
                        'Reconstruction L2 Loss/Train', total_train_l2_loss, iter_idx)

                    with torch.no_grad():
                        self.model.eval()
                        total_val_l2_loss = 9999
                        total_missing_total = 0

                        val_iter_idx = 0
                        for local_batch, local_label_dict in tqdm(self.val_loader, desc="Validating", leave=False):
                            val_iter_idx += 1
                            impute_out = self.model(local_batch)

                            val_l2_loss, missing_total = l2_loss(impute_out.to(torch.device(f"cuda:{self.gpu_list[0]}")),
                                                                 local_label_dict["target_seq"].to(torch.device(f"cuda:{self.gpu_list[0]}")))
                            total_val_l2_loss += val_l2_loss.item()
                            total_missing_total += missing_total

                        self.model.train()
                    total_val_l2_loss /= total_missing_total
                    writer.add_scalar(
                        'Reconstruction L2 Loss/Val', total_val_l2_loss, iter_idx)

                    printlog(
                        f'{datetime.now().strftime("%d/%m/%Y %H:%M")} | iter_idx: {iter_idx} \nTrain L2 Loss: {total_train_l2_loss:.8f} \nVal L2 Loss:{total_val_l2_loss:.8f} \n',
                        self.ckpt_path)

                    state = {'iter': iter_idx,
                             'state_dict': self.model.state_dict(),
                             'optimizer': self.optimizer.state_dict(),
                             "l2valloss": total_val_l2_loss}
                    iter_check_path = os.path.join(
                        self.ckpt_path, "iter_" + str(iter_idx))

                    os.mkdir(iter_check_path)
                    torch.save(state, os.path.join(
                        iter_check_path, "iter_" + str(iter_idx) + ".pkl"))

                    if total_val_l2_loss <= self.best_val_loss:
                        try:
                            os.remove(os.path.join(self.ckpt_path,
                                      "iter_best", "iter_best.pkl"))
                        except OSError:
                            pass
                        torch.save(state, os.path.join(
                            self.ckpt_path, "iter_best", "iter_best.pkl"))
                        self.best_val_loss = total_val_l2_loss

                    total_train_l2_loss = 0
                    total_missing_total = 0

        writer.close()


class dataset(torch.utils.data.Dataset):
    def __init__(self, waveforms, imputation_dict=None,
                 missingness_config=None,
                 type=None):

        self.waveforms = waveforms
        self.imputation_dict = imputation_dict
        self.missingness_config = missingness_config

        if "real" in missingness_config.miss_type:
            if missingness_config.miss_type == "miss_realppg":
                tuples_path = os.path.join(
                    "data", "missingness_patterns", f"missing_ppg_{type}.csv")
            elif missingness_config.miss_type == "miss_realecg":
                tuples_path = os.path.join(
                    "data", "missingness_patterns", f"missing_ecg_{type}.csv")

            with open(tuples_path, 'r') as read_obj:
                csv_reader = csv.reader(read_obj)
                self.list_of_miss = list(csv_reader)

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        X = torch.clone(self.waveforms[idx, :, :])
        X_original = torch.clone(X)
        y = np.empty(X.shape, dtype=np.float32)
        y[:] = np.nan
        y = torch.from_numpy(y)

        # using real mHealth missingness patterns
        if "real" in self.missingness_config.miss_type:
            miss_idx = np.random.randint(len(self.list_of_miss))
            miss_vector = miss_tuple_to_vector(self.list_of_miss[miss_idx])
            y[np.where(miss_vector == 0)] = X[np.where(miss_vector == 0)]
            X[np.where(miss_vector == 0)] = 0
        # using simulated extended missingness
        elif self.missingness_config.miss_type == "miss_extended":
            amt = self.missingness_config.miss
            start_impute = np.random.randint(
                0, X.shape[0]-amt)
            y[start_impute:start_impute+amt,
                :] = X[start_impute:start_impute+amt, :]
            X[start_impute:start_impute+amt, :] = 0
        # using simulated transient missingness
        elif self.missingness_config.miss_type == "miss_transient":
            window = self.missingness_config.miss["wind"]
            probability = self.missingness_config.miss["prob"]
            for j in range(0, X.shape[0], window):
                rand = np.random.random_sample()
                if rand <= probability:
                    if X.shape[0]-j < window:
                        incr = X.shape[0]-j
                    else:
                        incr = window
                    y[j:j+incr, :] = X[j:j+incr, :]
                    X[j:j+incr, :] = 0

        y_dict = {"target_seq": y,
                  "original": X_original,
                  "name": idx}

        if self.imputation_dict:
            y_dict["target_seq"] = self.imputation_dict["target_seq"][idx]

        return X, y_dict


def miss_tuple_to_vector(listoftuples):
    def onesorzeros_vector(miss_tuple):
        miss_tuple = literal_eval(miss_tuple)
        if miss_tuple[0] == 0:
            return np.zeros(miss_tuple[1])
        elif miss_tuple[0] == 1:
            return np.ones(miss_tuple[1])

    miss_vector = onesorzeros_vector(listoftuples[0])
    for i in range(1, len(listoftuples)):
        miss_vector = np.concatenate(
            (miss_vector, onesorzeros_vector(listoftuples[i])))
    miss_vector = np.expand_dims(miss_vector, 1)
    return miss_vector


def l2_loss(logits, target):
    logits_temp = torch.clone(logits)
    target_temp = torch.clone(target)

    logits_temp[torch.isnan(target)] = 0
    target_temp[torch.isnan(target)] = 0
    difference = torch.square(logits_temp - target_temp)

    l2_loss = torch.sum(difference)
    missing_total = torch.sum(~torch.isnan(target))

    return l2_loss, missing_total


def printlog(line, path="", type="a"):
    print(line)
    with open(os.path.join(path, 'log.txt'), type) as file:
        file.write(line+'\n')

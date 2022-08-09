from ..utils import utils
import os
import pickle
import pandas as pd
import numpy as np
import multiprocessing
from itertools import repeat
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class SCP_Experiment():
    '''
        Experiment on SCP-ECG statements. All experiments based on SCP are performed and evaluated the same way.
    '''

    def __init__(self, experiment_name, task, datafolder, outputfolder, models, sampling_frequency=100, min_samples=0, train_fold=8, val_fold=9, test_fold=10, folds_type='strat'):
        self.models = models
        self.min_samples = min_samples
        self.task = task
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.folds_type = folds_type
        self.experiment_name = experiment_name
        self.outputfolder = outputfolder
        self.datafolder = datafolder
        self.sampling_frequency = sampling_frequency

        # create folder structure if needed
        if not os.path.exists(os.path.join(self.outputfolder, self.experiment_name)):
            os.makedirs(os.path.join(self.outputfolder, self.experiment_name))
            if not os.path.exists(os.path.join(self.outputfolder, self.experiment_name+'/results/')):
                os.makedirs(os.path.join(self.outputfolder, self.experiment_name+'/results/'))
            if not os.path.exists(os.path.join(self.outputfolder, self.experiment_name+'/models/')):
                os.makedirs(os.path.join(self.outputfolder, self.experiment_name+'/models/'))
            if not os.path.exists(os.path.join(self.outputfolder, self.experiment_name+'/data/')):
                os.makedirs(os.path.join(self.outputfolder, self.experiment_name+'/data/'))

    def prepare(self, modelfolder=None, channel=None, data=None):
        print(f"Model Folder {modelfolder}")

        if data is None:
            self.data, self.raw_labels = utils.load_dataset(self.datafolder)
            if channel is not None:
                self.data = np.expand_dims(self.data[:,:,channel], axis=-1)

            # flatten mode align and bound
            X_flat = self.data.reshape(self.data.shape[0], -1)
            import warnings
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
            hist_out = np.apply_along_axis(lambda a: np.histogram(a, bins=50), 1, X_flat) # means we are applying function on this variable
            hist = hist_out[:, 0]
            bin_edges = hist_out[:, 1]
            # for i in range(X_flat.shape[1]):
            #     hist, bin_edges = np.histogram(X_flat[i], bins=50)
            
            def find_mode(hist, bin_edges):

                max_idx = np.argwhere(hist == np.max(hist))[0]
                mode = np.mean([bin_edges[max_idx], bin_edges[1+max_idx]])
                
                return mode
                
            modes = np.vectorize(find_mode)(hist, bin_edges)
            self.data = self.data - np.expand_dims(modes, axis = (1,2))
            max_val = np.amax(np.abs(X_flat), axis = 1, keepdims=True)
            self.data = self.data / np.expand_dims(max_val, axis = 2)/1
        else:
            # Load imputed PTB-XL data
            self.data = data
            self.raw_labels = pd.read_csv(os.path.join(self.datafolder, "ptbxl_database.csv"))
            if isinstance(self.test_fold, list):
                self.raw_labels = self.raw_labels[[True if label in self.test_fold else False for label in self.raw_labels.strat_fold]]
            else:
                self.raw_labels = self.raw_labels[self.raw_labels.strat_fold == self.test_fold]



        # Preprocess label data

        self.labels = utils.compute_label_aggregations(self.raw_labels, self.task, self.datafolder)
        


        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples)


        self.input_shape = self.data[0].shape
        
        # 10th fold for testing
        if isinstance(self.test_fold, list):
            self.X_test = self.data[[True if label in self.test_fold else False for label in self.labels.strat_fold]]
            self.y_test = self.Y[[True if label in self.test_fold else False for label in self.labels.strat_fold]]
        else:
            self.X_test = self.data[self.labels.strat_fold == self.test_fold]
            self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        if isinstance(self.train_fold, list):
            self.X_train = self.data[[True if label in self.train_fold else False for label in self.labels.strat_fold]]
            self.y_train = self.Y[[True if label in self.train_fold else False for label in self.labels.strat_fold]]
        else: 
            self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
            self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # import pdb; pdb.set_trace()

        # Preprocess signal data
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, os.path.join(self.outputfolder, self.experiment_name, 'data'),
        modelfolder=modelfolder)
        self.n_classes = self.y_train.shape[1]

        # save train and test labels
        self.y_train.dump(os.path.join(self.outputfolder, self.experiment_name+ '/data/y_train.npy'))
        self.y_val.dump(os.path.join(self.outputfolder, self.experiment_name+ '/data/y_val.npy'))
        self.y_test.dump(os.path.join(self.outputfolder, self.experiment_name+ '/data/y_test.npy'))


    def perform(self, modelfolder=None):

        for model_description in self.models:
            print("doing this model")
            print(model_description)
            modelname = model_description['modelname']
            modeltype = model_description['modeltype']
            modelparams = model_description['parameters']

            mpath = os.path.join(self.outputfolder, self.experiment_name+'/models/'+modelname+'/')
            # create folder for model outputs
            if not os.path.exists(mpath):
                os.makedirs(mpath)
            if not os.path.exists(mpath+'results/'):
                os.makedirs(mpath+'results/')

            n_classes = self.Y.shape[1]
            # import pdb; pdb.set_trace()
            # load respective model
            if modeltype == 'WAVELET':
                from ..models.wavelet import WaveletModel
                model = WaveletModel(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            elif modeltype == "fastai_model":
                from ..models.fastai_model import fastai_model
                model = fastai_model(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            elif modeltype == "YOUR_MODEL_TYPE":
                # YOUR MODEL GOES HERE!
                from ..models.your_model import YourModel
                model = YourModel(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            else:
                assert(True)
                break

            # fit model
            # import pdb; pdb.set_trace()
            if not modelfolder:
                model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
                # predict and dump
                # model.predict(self.X_train).dump(mpath+'y_train_pred.npy')
                # model.predict(self.X_val).dump(mpath+'y_val_pred.npy')
                model.predict(self.X_test).dump(mpath+'y_test_pred.npy') #aggregrating predictions
            else:
                model.predict(self.X_test, modelfolder).dump(mpath+'y_test_pred.npy')


    def evaluate(self, n_bootstraping_samples=10000, n_jobs=20, bootstrap_eval=False, dumped_bootstraps=True):
        # get labels
        y_train = np.load(os.path.join(self.outputfolder, self.experiment_name+'/data/y_train.npy'), allow_pickle=True)
        #y_val = np.load(self.outputfolder+self.experiment_name+'/data/y_val.npy', allow_pickle=True)
        y_test = np.load(os.path.join(self.outputfolder, self.experiment_name+'/data/y_test.npy'), allow_pickle=True)

        # if bootstrapping then generate appropriate samples for each
        if bootstrap_eval:
            if not dumped_bootstraps:
                #train_samples = np.array(utils.get_appropriate_bootstrap_samples(y_train, n_bootstraping_samples))
                test_samples = np.array(utils.get_appropriate_bootstrap_samples(y_test, n_bootstraping_samples))
                #val_samples = np.array(utils.get_appropriate_bootstrap_samples(y_val, n_bootstraping_samples))
            else:
                test_samples = np.load(os.path.join(self.outputfolder, self.experiment_name+'/test_bootstrap_ids.npy'), allow_pickle=True)
        else:
            #train_samples = np.array([range(len(y_train))])
            test_samples = np.array([range(len(y_test))])
            #val_samples = np.array([range(len(y_val))])

        # store samples for future evaluations
        #train_samples.dump(self.outputfolder+self.experiment_name+'/train_bootstrap_ids.npy')
        test_samples.dump(os.path.join(self.outputfolder, self.experiment_name+'/test_bootstrap_ids.npy'))
        #val_samples.dump(self.outputfolder+self.experiment_name+'/val_bootstrap_ids.npy')

        # iterate over all models fitted so far
        for m in sorted(os.listdir(os.path.join(self.outputfolder, self.experiment_name+'/models'))):
            print(m)
            mpath = os.path.join(self.outputfolder, self.experiment_name+'/models/'+m+'/')
            rpath = os.path.join(self.outputfolder, self.experiment_name+'/models/'+m+'/results/')

            # load predictions
            # y_train_pred = np.load(mpath+'y_train_pred.npy', allow_pickle=True)
            #y_val_pred = np.load(mpath+'y_val_pred.npy', allow_pickle=True)
            y_test_pred = np.load(mpath+'y_test_pred.npy', allow_pickle=True)

            if self.experiment_name == 'exp_ICBEB':
                # compute classwise thresholds such that recall-focused Gbeta is optimized
                thresholds = utils.find_optimal_cutoff_thresholds_for_Gbeta(y_train, y_train_pred)
            else:
                thresholds = None

            pool = multiprocessing.Pool(n_jobs)

            # tr_df = pd.concat(pool.starmap(utils.generate_results, zip(train_samples, repeat(y_train), repeat(y_train_pred), repeat(thresholds))))
            # tr_df_point = utils.generate_results(range(len(y_train)), y_train, y_train_pred, thresholds)
            # tr_df_result = pd.DataFrame(
            #     np.array([
            #         tr_df_point.mean().values, 
            #         tr_df.mean().values,
            #         tr_df.quantile(0.05).values,
            #         tr_df.quantile(0.95).values]), 
            #     columns=tr_df.columns,
            #     index=['point', 'mean', 'lower', 'upper'])

            te_df = pd.concat(pool.starmap(utils.generate_results, zip(test_samples, repeat(y_test), repeat(y_test_pred), repeat(thresholds))))
            te_df_point = utils.generate_results(range(len(y_test)), y_test, y_test_pred, thresholds)
            te_df_result = pd.DataFrame(
                np.array([
                    te_df_point.mean().values, 
                    te_df.mean().values,
                    te_df.quantile(0.05).values,
                    te_df.quantile(0.95).values]), 
                columns=te_df.columns, 
                index=['point', 'mean', 'lower', 'upper'])

            # val_df = pd.concat(pool.starmap(utils.generate_results, zip(val_samples, repeat(y_val), repeat(y_val_pred), repeat(thresholds))))
            # val_df_point = utils.generate_results(range(len(y_val)), y_val, y_val_pred, thresholds)
            # val_df_result = pd.DataFrame(
            #     np.array([
            #         val_df_point.mean().values, 
            #         val_df.mean().values,
            #         val_df.quantile(0.05).values,
            #         val_df.quantile(0.95).values]), 
            #     columns=val_df.columns, 
            #     index=['point', 'mean', 'lower', 'upper'])

            pool.close()

            # dump results
            #tr_df_result.to_csv(rpath+'tr_results.csv')
            #val_df_result.to_csv(rpath+'val_results.csv')
            te_df_result.to_csv(rpath+'te_results.csv')

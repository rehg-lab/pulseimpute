from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *

def main():

    # for impmodel in ["deepmvi","dcb883_emb256_layer2_test2", "conv9_emb256_layer2_test2", "van_emb256_posembed_layer2_test2", "brits_i_512", "naomi_step1", "naomi_step64", "naomi_step256"]:
    for impmodel in ["mean", "lininterp"]:

        for imptype in ["packet", "extended"]:
            if impmodel == "naomi_step64" and imptype == "extended":
                continue
            if impmodel == "naomi_step256" and imptype == "packet": # t/dcb883_emb256_layer2_test2_packet_ptbxl_extended_ptbxl
                continue
            impmodel_new = impmodel+"_"+imptype+"_ptbxl"
            if impmodel in ["mean", "lininterp"]:
                impmodel_new = impmodel 
            for per in [10,20,30,40,50]:
                datafolder = f'../../imputation/out/ptbxl_{imptype}_{per}percent/{impmodel_new}'
                # outputfolder = '../output_trainonimp/'
                outputfolder = '../output_noimputetrain/'
                models = [
                    # conf_fastai_inception1d
                     conf_fastai_xresnet1d101
                    ]
                ##########################################
                # STANDARD SCP EXPERIMENTS ON PTBXL
                ##########################################
                experiments = [
                    # (f'noimputetrain_{impmodel_new}_{per}{imptype}_subdiagnostic', 'subdiagnostic'),
                    # (f'noimputetrain_{impmodel_new}_{per}{imptype}_superdiagnostic', 'superdiagnostic'),
                    (f'noimputetrain_{impmodel_new}_{per}{imptype}_diagnostic', 'diagnostic'),
                    (f'noimputetrain_{impmodel_new}_{per}{imptype}_form', 'form'),
                    (f'noimputetrain_{impmodel_new}_{per}{imptype}_rhythm', 'rhythm'),
                ]
                print("this is datafolder " + datafolder)

                for name, task in experiments:

                    e = SCP_Experiment(name, task, datafolder, outputfolder, models, train_fold=4, val_fold=5, test_fold=[6,7,8,9,10])
                    e.prepare(modelfolder=outputfolder+"trainonfoldfour_"+task)
                    e.perform(modelfolder=outputfolder+"trainonfoldfour_"+task)
                    # e = SCP_Experiment(name, task, datafolder, outputfolder, models, train_fold=[6,7,8], val_fold=9, test_fold=10)
                    # e.prepare()
                    # e.perform()
                    e.evaluate()

                # generate greate summary table
                utils.generate_ptbxl_summary_table()
import numpy as np
import torch
import random

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    # num_threads_used = 56 // 8
    # print(f"Num Threads Used: {num_threads_used}")
    # torch.set_num_threads(num_threads_used)
    # os.environ["MP_NUM_THREADS"]=str(num_threads_used)
    # os.environ["OPENBLAS_NUM_THREADS"]=str(num_threads_used)
    # os.environ["MKL_NUM_THREADS"]=str(num_threads_used)
    # os.environ["VECLIB_MAXIMUM_THREADS"]=str(num_threads_used)
    # os.environ["NUMEXPR_NUM_THREADS"]=str(num_threads_used)


    random_seed(10, True)

    main()
#         experiments = [
#             (f'{impmodel}_{per}{imptype}_diagnostic', 'diagnostic'),
#             (f'{impmodel}_{per}{imptype}_form', 'form'),
#             (f'{impmodel}_{per}{imptype}_rhythm', 'rhythm'),
#         ]

The structure of these files in the repo is:

configs/
- Base_configs.py
- FFT_new_configs.py

I also provided the old version of the configs to show other model configs as well, but they would be fit to FFT_new_configs style if they existed. These include BDC_configs.py, Conv9..py, FFT_configs.py.

data/
PulseImputeData.py
PTBXLDataset.py
ECGMIMICDataset.py
PPGMIMICDataset.py

demo/
interactive_plotly.ipynb

experiment/
Miss_MIMICdata_Experiment.py
Miss_PTBXLdata_Experiment.py
PulseImputeExperiment.py

models/
models/ClassicalModel_Architecture/
- ClassicalModel_Wrapper.py
- FFT.py
models/TransformerModel_Architecture/
- BDCTransformer.py
- TransformerModel_Wrapper.py

utils/
- evaluate_imputation.py
- loss_mask.py
- random_seed.py
- visualize.py

utils/missingness/
- base_missingness.py
- extended_missingness.py
- mimic_missingness.py
- transient_missingness.py

In the main folder:
configs.yaml
run_experiment.py
pulseimpute.yml


There are 10 models in the repo, but I only show mostly FFT in the provided files. 
README for PulseImpute Data

We license this data under the Creative Commons Attribution 4.0 International, and it is 82 GB uncompressed. Please see the original paper, PulseImpute: A Novel Benchmark Task for Pulsative Physiological Signal Imputation, published in the 2022 Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks for more details on the dataset and intended use-case. The associated code repository for this challenge can be found here: https://github.com/rehg-lab/pulseimpute. 


The curated data and missingness patterns are organized as shown below: 

->/pulseimpute_data/
------> README.md
------> missingness_patterns/
-----------> mHealth_missing_ecg/
----------------> missing_ecg_train.csv
----------------> missing_ecg_val.csv
----------------> missing_ecg_test.csv
-----------> mHealth_missing_ppg/
----------------> missing_ppg_train.csv
----------------> missing_ppg_val.csv
----------------> missing_ppg_test.csv
------> waveforms/
-----------> mimic_ecg/
----------------> mimic_ecg_train.npy
----------------> mimic_ecg_val.npy
----------------> mimic_ecg_test.npy
----------------> MIMIC_III_ECG_filenames.txt
-----------> mimic_ppg/
----------------> mimic_ppg_train.npy
----------------> mimic_ppg_val.npy
----------------> mimic_ppg_test.npy
----------------> MIMIC_III_PPG_filenames.txt
-----------> ptbxl_ecg/
----------------> scp_statements.csv
----------------> ptbxl_database.csv
----------------> ptbxl_ecg.npy


The missingness patterns are stored in csv files, with each row as a list of tuples of size 2, which represent the binary missingness pattern time-series. The first item in the tuple corresponds to missing (0) or not missing (1) with the second entry corresponding to the length of samples (in 100 hz) that the missing or not missingness segment lasts. The waveform data is all stored as .npy files, with each row corresponding to a 100 hz waveform. In order to download a waveform individually, please see our code repository for a link to a dataset with each individual waveform saved as an individual file.


Each of MIMIC-III curated data and missingness patterns have been split into 80/10/10 training/validation/testing splits accordingly.  If we concatenate train, validation, and test .npy files in that order, each data index corresponds to the file name at the corresponding line in the MIMIC_III_ECG_filenames.txt or MIMIC_III_PPG_filenames.txt file. 


For the cardiac classification tasks on the PTB-XL data, the labels can be found in the ptbxl_database.csv file and the waveform data is the same as the original PTB-XL dataset.  We use the original paper's proposed splits to divide into 40/10/50 training/validation/testing splits. Imputation models and downstream cardiac classification models are trained with the aforementioned 40/10 split, with the classification model training on the clean non-imputed data. Then the classification model runs inference on the imputed test data in the 50 split to evaluate imputation quality. This large test split was done to allow for future work where the classification model trains directly on imputed data.
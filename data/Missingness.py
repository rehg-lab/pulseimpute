from abc import ABC, abstractmethod
import numpy as np
import csv

from ast import literal_eval

class MissingnessGenerator(ABC):

    @abstractmethod
    def apply(self, data):
        pass

class ExtendedMissingness(MissingnessGenerator):
    def __init__(self, percent):
        self.percent = percent

    def apply(self, data):
        total_len = data.shape[1]
        amt_impute = int(total_len * self.percent / 100)
        target = np.empty(data.shape, dtype=np.float32)
        target[:] = np.nan
        input_data = np.copy(data)

        for i in range(data.shape[0]):
            for j in range(data.shape[-1]):
                start_impute = np.random.randint(0, total_len - amt_impute)
                target[i, start_impute:start_impute+amt_impute, j] = data[i, start_impute:start_impute+amt_impute, j]
                input_data[i, start_impute:start_impute+amt_impute, j] = np.nan
                data[i, start_impute:start_impute+amt_impute, j] = 0

        return data, input_data, target

class TransientMissingness(MissingnessGenerator):
    def __init__(self, window, prob):
        self.window = window
        self.prob = prob

    def apply(self, data):
        total_len = data.shape[1]
        target = np.empty(data.shape, dtype=np.float32)
        target[:] = np.nan
        input_data = np.copy(data)

        for i in range(data.shape[0]):
            for start_impute in range(0, total_len, self.window):
                for j in range(data.shape[-1]):
                    if np.random.random() <= self.prob:
                        end_impute = min(start_impute + self.window, total_len)
                        target[i, start_impute:end_impute, j] = data[i, start_impute:end_impute, j]
                        input_data[i, start_impute:end_impute, j] = np.nan
                        data[i, start_impute:end_impute, j] = 0

        return data, input_data, target

class MIMICMissingness(MissingnessGenerator):
    def __init__(self, miss_tuples_path):
        self.miss_tuples = self.load_miss_tuples(miss_tuples_path)

    def load_miss_tuples(self, path):
        with open(path, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            return list(csv_reader)

    def miss_tuple_to_vector(self, listoftuples):
        def onesorzeros_vector(miss_tuple):
            miss_tuple = literal_eval(miss_tuple)
            if miss_tuple[0] == 0:
                return np.zeros(miss_tuple[1])
            elif miss_tuple[0] == 1:
                return np.ones(miss_tuple[1])

        miss_vector = onesorzeros_vector(listoftuples[0])
        for i in range(1, len(listoftuples)):
            miss_vector = np.concatenate((miss_vector, onesorzeros_vector(listoftuples[i])))
        return np.expand_dims(miss_vector, 1)

    def apply(self, data):
        target = np.empty(data.shape, dtype=np.float32)
        target[:] = np.nan
        input_data = np.copy(data)

        for iter_idx, waveform_idx in enumerate(range(0, data.shape[0], 4)):
            miss_idx = iter_idx % len(self.miss_tuples)
            miss_vector = self.miss_tuple_to_vector(self.miss_tuples[miss_idx])
            
            for i in range(min(4, data.shape[0] - waveform_idx)):
                target[waveform_idx + i, np.where(miss_vector == 0)[0]] = data[waveform_idx + i, np.where(miss_vector == 0)[0]]
                input_data[waveform_idx + i] *= miss_vector
                data[waveform_idx + i] *= miss_vector

        return data, input_data, target



class CustomMissingness(MissingnessGenerator):
    def __init__(self, custom_param):
        self.custom_param = custom_param

    def apply(self, data):
        ###
        # Implement some missingness logic here
        ###
        target = np.empty(data.shape, dtype=np.float32)
        target[:] = np.nan
        input_data = np.copy(data)

        ###
        # Custom missingness logic here
        ###

        return data, input_data, target


def apply_missingness(data, missingness_generator):
    return missingness_generator.apply(data)
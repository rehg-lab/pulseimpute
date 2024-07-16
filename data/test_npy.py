import numpy as np

load_file = 'data/data/custom_1split/asdfadsf.npy'
#load_file = 'out/out_test/custom_test/FFT_custom/original.npy'
data = np.load(load_file)

#reshaped_tensor = data.reshape(data.shape[0], data.shape[1])
#np.save(load_file, reshaped_tensor)

print(data.shape)

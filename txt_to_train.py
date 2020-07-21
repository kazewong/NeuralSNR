import h5py
import numpy as np

# Customize input and output here

outputname = './data/test_random.hdf5'
input_txt = './data/input_random.txt'
output_txt = './data/output_random.txt'

input_file = np.genfromtxt(input_txt)
output_file = np.genfromtxt(output_txt)

assert input_file.shape[0]==output_file.shape[0]
totalN = input_file.shape[0]
index = np.arange(totalN)
np.random.shuffle(index)

train_in = input_file[index][:int(totalN*8/10)]
valid_in = input_file[index][int(totalN*8/10):int(totalN*9./10)]
test_in = input_file[index][int(totalN*9./10):]
train_out = output_file[index][:int(totalN*8/10)]
valid_out = output_file[index][int(totalN*8/10):int(totalN*9/10)]
test_out = output_file[index][int(totalN*9/10):]

with h5py.File(outputname,'w') as f:
	train = f.create_group('Train')
	valid = f.create_group('Valid')
	test = f.create_group('Test')
	train.create_dataset('input',data=train_in)
	train.create_dataset('output',data=train_out)
	valid.create_dataset('input',data=valid_in)
	valid.create_dataset('output',data=valid_out)
	test.create_dataset('input',data=test_in)
	test.create_dataset('output',data=test_out)
	f.close()



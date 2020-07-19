import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class FunctionIO_DataSet(Dataset):
	def __init__(self,Input,Output):
		self.Input = torch.from_numpy(Input).float()
		self.Output = torch.from_numpy(Output).float()

	def __len__(self):
		return self.Input.size()[0]

	def __getitem__(self,idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		Input = self.Input[idx]
		Output = self.Output[idx]
		
		return Input,Output

class FunctionIO_Loader:

	def __init__(self,config):
		self.config = config

		data = h5py.File(config.data,mode='r')

		train = FunctionIO_DataSet(data['Train/input'][()],data['Train/output'][()])
		valid = FunctionIO_DataSet(data['Valid/input'][()],data['Valid/output'][()])
		test = FunctionIO_DataSet(data['Test/input'][()],data['Test/output'][()])
	
		
		self.train_loader = DataLoader(train,batch_size=config.batch_size,shuffle=True)
		self.valid_loader = DataLoader(valid,batch_size=config.batch_size,shuffle=False)
		self.test_loader = DataLoader(test,batch_size=config.batch_size,shuffle=False)

	def make_batch_plot(self):
		pass

	def finalize(self):
		pass

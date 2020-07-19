import math
import types
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(torch.nn.Module):
	def forward(self, x):
		return x * torch.sigmoid(x)

class MLP(nn.Module):
	def __init__(self, input_size,output_size,hidden_size,hidden_layers=5):
		super().__init__()
		layers = np.array([nn.Linear(hidden_size, hidden_size) for i in range(hidden_layers)])
		layers = np.insert(layers,np.arange(1,hidden_layers+1),Swish())
		self.output = nn.Sequential(
		nn.Linear(input_size, hidden_size), Swish(),
		*layers,
		nn.Linear(hidden_size, output_size))

	def forward(self, x):
		return self.output(x)


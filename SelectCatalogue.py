import argparse
import numpy as np
import torch
import h5py
from astropy.cosmology import Planck15 
from scipy.interpolate import interp1d
import os

parser = argparse.ArgumentParser(description='Hierarchial Bayesian inference with Gaussian process regession to interpolate the hyper parameter space.')
parser.add_argument('--CatalogueList',type=str,help='txt file containing all catalogue name',required=True)
parser.add_argument('--Network',type=str,help='Network for calculating SNR',required=True)
parser.add_argument('--Threshold',type=float,help='SNR threshold for detection',default=8)

args = parser.parse_args()
catalogue_list = args.CatalogueList
network = args.Network
threshold = args.Threshold

# ===================================================
# Load model and detector response function
# ===================================================

model = torch.jit.load(network,map_location='cpu')
P_omega_data = np.genfromtxt('./data/Pw_single.dat').T
P_omega = interp1d(P_omega_data[0],P_omega_data[1],bounds_error=False,fill_value=(1,0))
z_axis = np.linspace(0,10,10000)
d_axis = Planck15.luminosity_distance(z_axis).value
dL_interp = interp1d(z_axis,d_axis)

# ===================================================
# Define model calls
# ===================================================

@torch.no_grad()
def get_detect(M1M2s1s2,z,threshold):
	total_size = M1M2s1s2.shape[0]
	dL = dL_interp(z)
	snr = model(M1M2s1s2)[:,0].numpy()*100/dL
	pdet = P_omega(threshold/snr)
	detect_size = np.sum(pdet[snr>threshold])
	return snr,pdet,M1M2s1s2[snr>threshold],detect_size/total_size

catalogue_file = open(catalogue_list,'r')
for catalogue in catalogue_file:
	output_file = catalogue[:-5]+'_detect'
	data = torch.from_numpy(np.genfromtxt(catalogue[:-1])).float()
	output = get_detect(data[:,:4],data[:,-1],threshold)
	np.savez(output_file,snr=output[0],pdet=output[1],select_sample=output[2],fraction=output[3],model=network)

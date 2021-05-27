# Test Learning MNIST


import sys
sys.path.append('src/')
import torch
import rbm
import torchvision.datasets as datasets
import numpy as np

device = torch.device("cpu")
# Uncomment if you have a GPU
# device = torch.device("cuda:0")
dtype = torch.float
# Set the number of threads to bb used (it only affect the CPU)
torch.set_num_threads(12)

Ns = 10000
mnist_trainset = datasets.MNIST('dataset/', train=True, download=True)
X = mnist_trainset.data[:Ns,:,:].reshape(Ns,28*28).float().to(device) / 255.0
X = (X > 0.3) * 1.0
X = X.t()

Nv = X.shape[0] # numbder of visible nodes
Nh = 50 # number of hidden nodes
lr = 0.01 # learning rate
NGibbs=10  # number of gibbs steps
n_mb = 50 # size of minibatches
n_pcd = n_mb # size of the negative chain

myRBM = rbm.RBM(num_visible=Nv,
				num_hidden=Nh,
				device=device,
				lr=lr,
				gibbs_steps=NGibbs,
				UpdCentered=True,
				mb_s=n_mb,
				num_pcd=n_pcd)
			
myRBM.SetVisBias(X) # initialize the visible biases
myRBM.ResetPermChainBatch = True # Put False for PCD, False give Rdm
stamp = 'test'
# stamp='LongRUNExMC_TEMP_'+str(Temp)+'_MNIST_Nh'+str(Nh)+'_lr'+str(lr)+'_l2'+str(l2)+'_NGibbs'+str(NGibbs)
myRBM.file_stamp = stamp	
base = 1.7
v = np.array([0,1],dtype=int)
allm = np.append(np.array(0),base**np.array(list(range(30))))
for k in range(30):
	for m in allm:
		v = np.append(v,int(base**k)+int(m))

v = np.array(list(set(v)))
v = np.sort(v)
myRBM.list_save_time = v

ep_max = 1000
fq_msr_RBM = 10
myRBM.list_save_rbm = np.arange(1,ep_max,fq_msr_RBM)	

myRBM.fit(X,ep_max=ep_max)
	

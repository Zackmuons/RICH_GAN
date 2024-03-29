import sys
import numpy as np
import random
import math
import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import pickle


##################################  Load Data ########################################

real_data = torch.load("master_tensor.pt")
test_data = real_data[5518:6036, :]

print(real_data.shape)





targets_rnc = 4
conds_rnc = 6


test_conds_rnc = test_data[:, :conds_rnc]#.reshape(-1,1)
test_targets_rnc = test_data[:,conds_rnc:conds_rnc+4]
test_conds_ring = test_targets_rnc
test_hits= test_data[:,conds_rnc+4:]
print(test_hits[:, 0])
test_size = noise_size = test_data.shape[0]
n_hits_ints = test_data[:,-1]

################################### Define Model Architecture ###########################

class Discriminator_ring(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
		nn.Linear(5, 256),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Dropout(0.3),
		nn.Linear(256, 128),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Dropout(0.3),
		nn.Linear(128, 1),
		nn.Sigmoid(),
		)

	def forward(self, x):
		output = self.model(x)
		return output

discriminator_ring = Discriminator_ring()

#added extra layer to generator, maybe needs it?
class Generator_ring(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(5, 32),
			nn.ReLU(),
			nn.Linear(32, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
			)

	def forward(self, x):
		output = self.model(x)
		return output
    
generator_ring = Generator_ring()


class Discriminator_rnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
        nn.Linear(targets_rnc + conds_rnc, 128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 128),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Dropout(0.3),
        # nn.Linear(128, 128),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Dropout(0.3),
        # nn.Linear(128, 128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 128),
        nn.Linear(128, 1),
        nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

discriminator_rnc = Discriminator_rnc()

class Generator_rnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(targets_rnc+conds_rnc, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, targets_rnc),
            )

    def forward(self, x):
        output = self.model(x)
        return output

generator_rnc = Generator_rnc()

############################## Load Models ##################################
# Load discriminator and generator models for RnCGAN
rnc_models = torch.load('rnc_1950000_roc_0.57.pth', map_location=torch.device('cpu'))
discriminator_rnc.load_state_dict(rnc_models['discriminator_state_dict'])
generator_rnc.load_state_dict(rnc_models['generator_state_dict'])

# Load discriminator and generator models for RingGAN

discriminator_ring.load_state_dict(torch.load('D_ringgancenters.pth', map_location=torch.device('cpu')))
generator_ring.load_state_dict(torch.load('G_ringgancenters.pth', map_location=torch.device('cpu')))

###################### Generate RNC #########################################
noise_vector_rnc = torch.randn(test_size, targets_rnc)
latent_space_vector_rnc = torch.cat([test_conds_rnc, noise_vector_rnc], dim =1)
print(latent_space_vector_rnc.shape)
rnc_gen = generator_rnc(latent_space_vector_rnc).detach()
print(rnc_gen)
##################### Generate HITS #######################################
noise_vector_ring = torch.randn(test_size, 2)
latent_space_vector_ring = torch.cat([noise_vector_ring, rnc_gen[:,:-1]], dim = 1)
print(latent_space_vector_ring.shape)
ring_gen = generator_ring(latent_space_vector_ring).detach()

print(ring_gen)


###################### displace hits ###########################################

###################### plot stuff ###########################################
plt.figure(figsize = (8,8))
plt.plot(ring_gen[:,0],ring_gen[:,1],'.')
plt.title('Generated Hits')
plt.show()

# all_hits = []
# for i in range(test_size):
#     for i in range():

hit_number = 500
plt.figure(figsize = (8,8))
plt.plot(test_hits[hit_number,:88],test_hits[hit_number,88:-1],'.')
plt.title('Real Hits')
plt.show()







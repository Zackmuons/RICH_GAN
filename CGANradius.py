# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:49:57 2023

@author: Zack Amos
"""

# Import WassersteinLoss from PyTorch
#from torch.nn.modules.loss import WassersteinLoss

# Define the Wasserstein loss function
#wasserstein_loss = WassersteinLoss()

import numpy as np
import math
import torch
from torch import nn
import matplotlib.pyplot as plt
import time

torch.set_default_dtype(torch.float32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('torch version:',torch.__version__)
print('device:', device)

train_size = 1024


radii = [0.1, 0.2, 0.25, 0.3 , 0.4]
radii = torch.tensor(radii).to(device)

no_r = len(radii)
points = 5

# returning to x,y coordinates

def makevector(size, radii, points, real = False, just_r = False):
	#set of alternating radii
	cond = np.ones(size, np.float32)*[radii[i%no_r] for i in range(size)]
	cond = cond[:,None]
	
	#print(cond)
	
	cond = torch.tensor(cond).to(device)
	
	if just_r :
		return cond
		
	
	if not real:
		# the random part
		latent = torch.randn(size,2*points).to(device)
	
	if real:
		#set up variables
		theta = torch.rand(size,points,2)*2.*math.pi
		latent = torch.zeros(size,points,2)
		
		sin = torch.sin(theta)
		cos = torch.cos(theta)
		
		#hopefully all this bs works
		latent[:,:,0] = cos[:,:,0]
		latent[:,:,1] = sin[:,:,0]
		#print(latent)
		latent = latent.view(size, 2*points)
		#print(latent)
		latent = torch.multiply(latent,cond)
		#print(latent)
		
	
	if not just_r:
		vector = torch.cat((latent,cond),1)
		#print(vector)
		return vector.to(device)

#define our train set of sets of 5 points with correponding radii
train_set = makevector(train_size, radii, points, real = True)



#batch that shizzle
batch_size = 32

#train_loader = train_set.view(32, batch_size, 2*points+1)
#print(train_loader)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=False
)
print(train_loader)


# want to optimise these also add cuda possibilities
class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
		nn.Linear(points*2 + 1, 256),
		nn.ReLU(),
		nn.Dropout(0.3),
		nn.Linear(256, 128),
		nn.ReLU(),
		nn.Dropout(0.3),
		nn.Linear(128, 64),
		nn.ReLU(),
		nn.Dropout(0.3),
		nn.Linear(64, 1),
		nn.Sigmoid(),
		)

	def forward(self, x):
		output = self.model(x)
		return output

discriminator = Discriminator().to(device)

#added extra layer to generator, maybe needs it?
class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(points*2 + 1, 16),
			nn.ReLU(),
			nn.Linear(16, 32),
			nn.ReLU(),
			nn.Linear(32, points*2+1),
			)

	def forward(self, x):
		output = self.model(x)
		return output

generator = Generator().to(device)

# define hyperparameters

lr = 0.00005
num_epochs = 1001
loss_function = nn.BCELoss()


loss_list = np.empty((0,3))


optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

counter = 0


# Load discriminator and generator models

discriminator.load_state_dict(torch.load('D_CGAN1.pth'))
generator.load_state_dict(torch.load('G_CGAN1.pth'))



for epoch in range(num_epochs):
	
	t = time.time()
	for n, real_samples in enumerate(train_loader):
		# make this stuff into functions i reckon
		# Data for training the discriminator
		#print(real_samples)
		latent_space_samples = makevector(batch_size, radii, points, real = False )# normalised???!!
		#print(latent_space_samples)
		#generate samples
		generated_samples = generator(latent_space_samples)
		generated_samples_labels = torch.zeros((batch_size,1))
		real_samples_labels = torch.ones((batch_size,1))
		
		
		all_samples = torch.cat((real_samples, generated_samples))
		all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))


		"""
		#wasserstein bit
        # Training the discriminator
		discriminator.zero_grad()
		output_discriminator = discriminator(all_samples)

        # Calculate Wasserstein loss and optimize
		loss_discriminator = -output_discriminator.mean()  # Negative since it's a minimization problem
		loss_discriminator.backward()
		optimizer_discriminator.step()

        # Data for and training of the generator
		latent_space_samples = makevector(batch_size, radii, points, real=False)
		generated_samples = generator(latent_space_samples)
		output_discriminator_generated = discriminator(generated_samples)

        # Calculate Wasserstein loss for the generator
		loss_generator = -output_discriminator_generated.mean()  # Negative since it's a minimization problem
		loss_generator.backward()
		optimizer_generator.step()
		


		"""
		# Training the discriminator
		discriminator.zero_grad()
		output_discriminator = discriminator(all_samples)
		
		print([torch.mean(output_discriminator[:15]),torch.mean(output_discriminator[:-15])])
		#calc loss and optimise
		

		loss_discriminator = loss_function(output_discriminator, all_samples_labels)
		loss_discriminator.backward()
		optimizer_discriminator.step()

		# Data for and training of the generator

		generator.zero_grad()
		latent_space_samples = makevector(batch_size,radii,points, real = False )
		generated_samples = generator(latent_space_samples)
		#output_discriminator_generated = discriminator(torch.cat((generated_samples, the_r),1))
		output_discriminator_generated = discriminator(generated_samples)

		#calc loss and optimise
		loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
		loss_generator.backward()
		optimizer_generator.step()
		
		
		loss_g = loss_generator.detach().numpy()
		loss_d = loss_discriminator.detach().numpy()
		loss_list = np.append(loss_list, [[counter, loss_g, loss_d]], axis=0)
		counter +=1
		# Show loss

		if epoch % 100 == 0 and n == batch_size - 1:
			e = epoch + 1200
			print(f"Epoch: {e} loss D: {loss_discriminator}")
			print(f"Epoch: {e} loss G: {loss_generator}")

			
			plt.savefig(f'testset_{epoch}.png')
			plt.close('all')
			
			
			plt.figure(figsize=(16,16))
			plt.subplot(2,2,1)
			#plt.title.set_text("Discriminator loss")
			plt.plot(loss_list[:,0], loss_list[:,1])
			plt.subplot(2,2,2)
			#plt.title.set_text("Generator loss")
			plt.plot(loss_list[:,0], loss_list[:,2])
			
			
			r1 = torch.ones(10)*radii[0]
			r1 = r1[:,None]
			r2 = torch.ones(10)*radii[3]
			r2 = r2[:,None]
			latent_space_samples = torch.randn(10, points*2)
			v1 = torch.cat((latent_space_samples, r1),1)
			v2 = torch.cat((latent_space_samples, r2),1)
			gener1 = generator(v1)
			gener2 = generator(v2)
			gener1 = gener1.detach()
			gener2 = gener2.detach()
			
			gener1 = gener1[:,:-1]
			gener2 = gener2[:,:-1]
			
			plt.subplot(2,2,3)
			plt.xlim(-0.5,0.5)
			plt.ylim(-0.5,0.5)
			
			gener1 = gener1.reshape(10*points,2)
			plt.plot(gener1[:,0], gener1[:,1] , ".")
			
			
			plt.subplot(2,2,4)
			plt.xlim(-0.5,0.5)
			plt.ylim(-0.5,0.5)
			gener2 = gener2.reshape(10*points,2)
			plt.plot(gener2[:,0],gener2[:,1] , ".")
			
			plt.savefig(f'testset_{e}.png')
			plt.close('all')
		
		if epoch == 1000:
			

			# Create a dictionary to save GAN information
			gan_info = {
					'discriminator_state_dict': discriminator.state_dict(),
					'generator_state_dict': generator.state_dict(),
					}
			# Save discriminator and generator models

			torch.save(discriminator.state_dict(), 'D_CGAN1.pth')
			torch.save(generator.state_dict(), 'G_CGAN1.pth')
			# Save GAN information
			torch.save(gan_info, 'gan_info_CGAN1-2.pth')
			#print("Epoch: {} time: {}".format(epoch,time.time()-t))
		



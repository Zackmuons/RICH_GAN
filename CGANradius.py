# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:49:57 2023

@author: Zack Amos
"""

import numpy as np
import math
import torch
from torch import nn
import matplotlib.pyplot as plt
import time

train_size = 1024
halfsize = int(train_size/2)

signal1 = torch.zeros((halfsize,2))
signal2 = torch.zeros((halfsize,2))
theta = torch.rand(train_size)*2.*math.pi
r = 0.1
signal1[:,0] = r
signal1[:,1] = theta[:512]

signal2[:,0] = 2*r
signal2[:,1] = theta[:-512]

labels1 = torch.ones(halfsize,1)
labels2 = 2*torch.ones(halfsize,1)

radii = np.multiply([1,1.5,2,2.5,3,3.5,4],r)

signal = torch.cat((signal1, signal2))
labels = torch.cat((labels1, labels2))
signal = torch.cat((signal, labels),1)
# returning to x,y coordinates
"""
def makevectors(size, radii, real = True):
	cond = r*torch.ones(size)*[radii[i%len(radii)] for i in range(size)]
	noise = torch.randn(size,5,2)
	
	if real:
		for i in range(size):
			noise[i,]
			
	else:
		vector = torch.cat((noise,cond),)
	return vector
"""

train_set = [(signal[i], labels[i]) for i in range(train_size)]


#plt.plot(signal[:,0],signal[:,1],".")
#plt.show()

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
        nn.Linear(3, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.1),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    def forward(self, x):
        output = self.model(x)
        return output

discriminator = Discriminator()

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator()

lr = 0.0001
num_epochs = 201
loss_function = nn.BCELoss()

loss_list = np.empty((0,3))

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

counter = 0


# Load discriminator and generator models
discriminator.load_state_dict(torch.load('D_gen5K.pth'))
generator.load_state_dict(torch.load('G_gen5K.pth'))



for epoch in range(num_epochs):
    
    t = time.time()
    for n, (real_samples, _)  in enumerate(train_loader):
        
        # Data for training the discriminator
        radii = torch.cat((torch.ones(int(batch_size/2)),2*torch.ones(int(batch_size/2))))
        radii = radii[:,None]
        latent_space_samples = torch.randn((batch_size, 2))# normalised???!!
		#generate samples
        generated_samples = generator(torch.cat((latent_space_samples, radii),1))
        generated_samples_labels = torch.zeros((batch_size,1))
        real_samples_labels = torch.ones((batch_size,1))
        generated_samples = torch.cat((generated_samples, radii),1)
		
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
		#calc loss and optimise
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for and training of the generator
        #radii = torch.cat((torch.ones(int(batch_size/2)),2*torch.ones(int(batch_size/2))))
        #radii = radii[:,None]
        #latent_space_samples = torch.randn((batch_size, 2))
        generated_samples = generator(torch.cat((latent_space_samples, radii),1))
        output_discriminator_generated = discriminator(torch.cat((generated_samples, radii),1))
		#calc loss and optimise
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()
        
		
        loss_g = loss_generator.detach().numpy()
        loss_d = loss_discriminator.detach().numpy()
        loss_list = np.append(loss_list, [[counter, loss_g, loss_d]], axis=0)
        counter +=1
        # Show loss
       
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} loss D: {loss_discriminator}")
            print(f"Epoch: {epoch} loss G: {loss_generator}")
			
			
            
            plt.figure(figsize=(8,8))
            plt.subplot(2,2,1)
            #plt.title.set_text("Discriminator loss")
            plt.plot(loss_list[:,0], loss_list[:,1])
            plt.subplot(2,2,2)
            #plt.title.set_text("Generator loss")
            plt.plot(loss_list[:,0], loss_list[:,2])
            
            
            radii1 = torch.ones(batch_size)
            radii1 = radii1[:,None]
            radii2 = torch.ones(batch_size)*2
            radii2 = radii2[:,None]
            latent_space_samples = torch.randn((batch_size, 2))
            gener1 = generator(torch.cat((latent_space_samples, radii1),1))
            gener2 = generator(torch.cat((latent_space_samples, radii2),1))
            gener1 = gener1.detach()
            gener2 = gener2.detach()
            
            plt.subplot(2,2,3)
            plt.xlim(-0.5,0.5)
            plt.ylim(-0.5,0.5)
            sin = torch.sin(gener1[:, 1])
            cos = torch.cos(gener1[:, 1])
            x = torch.multiply(gener1[:, 0],sin)
            y = torch.multiply(gener1[:, 0],cos)
            plt.plot(x,y , ".")
            
            
            plt.subplot(2,2,4)
            plt.xlim(-0.5,0.5)
            plt.ylim(-0.5,0.5)
            sin = torch.sin(gener2[:, 1])
            cos = torch.cos(gener2[:, 1])
            x = torch.multiply(gener2[:, 0],sin)
            y = torch.multiply(gener2[:, 0],cos)
            plt.plot(x,y , ".")
                
            plt.savefig(f'CGANradius_{epoch}.png')
            plt.close('all')
            
        if epoch == 201:
			# Save discriminator and generator models
            torch.save(discriminator.state_dict(), 'D_gen5K.pth')
            torch.save(generator.state_dict(), 'G_gen5K.pth')
			# Create a dictionary to save GAN information
            gan_info = {
			        'discriminator_state_dict': discriminator.state_dict(),
			        'generator_state_dict': generator.state_dict(),
            }
			# Save discriminator and generator models
            torch.save(discriminator.state_dict(), 'D_gen5K.pth')
            torch.save(generator.state_dict(), 'G_gen5K.pth')
			# Save GAN information
            torch.save(gan_info, 'gan_info_5K.pth')
    #print("Epoch: {} time: {}".format(epoch,time.time()-t))




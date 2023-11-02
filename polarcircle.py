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

signal = torch.cat((signal1, signal2))
labels = torch.cat((labels1, labels2))
signal = torch.cat((signal, labels),1)

train_set = [torch.cat((signal[i], labels[i])) for i in range(train_size)]


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
num_epochs = 101
loss_function = nn.BCELoss()

loss_list = np.empty((0,3))

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

counter = 0




for epoch in range(num_epochs):
    print(epoch)
    loader = enumerate(train_loader)
    loaderdict = dict(loader)
    for n, (real_samples, real_samples_labels)  in enumerate(train_loader):
        print(n)
        # Data for training the discriminator
        radii = torch.cat((torch.ones(int(batch_size/2)),2*torch.ones(int(batch_size/2))))
        latent_space_samples = torch.randn((batch_size, 2))
		#generate samples with 
        generated_samples = generator(torch.cat((latent_space_samples, [radii]),1))
        generated_samples_labels = torch.zeros((batch_size,1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        # Training the discriminator on real samples
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for and training of the generator
        radii = torch.cat((torch.ones(int(batch_size/2)),2*torch.ones(int(batch_size/2))))
        latent_space_samples = torch.randn((batch_size, 2))
        generated_samples = generator(torch.cat((latent_space_samples, [radii]),1))
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()
        
        
        loss_list = np.append(loss_list, [[counter, loss_generator.detach().numpy(), loss_discriminator.detach().numpy()]], axis=0)
        counter +=1
        # Show loss
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} loss D: {loss_discriminator}")
            print(f"Epoch: {epoch} loss G: {loss_generator}")
            
            plt.figure(figsize=(8,8))
            plt.subplot(2,2,1)
            plt.plot(loss_list[:,0], loss_list[:,1])
            plt.subplot(2,2,2)
            plt.plot(loss_list[:,0], loss_list[:,2])
            
            
            radii = torch.cat((torch.ones(int(batch_size/2)),2*torch.ones(int(batch_size/2))))
            latent_space_samples = torch.randn((batch_size, 2))
            generated_samples = generator(torch.cat((latent_space_samples, [radii]),1))
            generated_samples = generated_samples.detach()
            
            plt.subplot(2,2,3)
            plt.xlim(-0.25,0.25)
            plt.ylim(-0.25,0.25)
            sin = torch.sin(signal[:, 1])
            cos = torch.cos(signal[:, 1])
            x = torch.multiply(signal[:, 0],sin)
            y = torch.multiply(signal[:, 0],cos)
            plt.plot(x,y , ".")
            
            
            plt.subplot(2,2,4)
            plt.xlim(-0.25,0.25)
            plt.ylim(-0.25,0.25)
            sin = torch.sin(generated_samples[:, 1])
            cos = torch.cos(generated_samples[:, 1])
            x = torch.multiply(generated_samples[:, 0],sin)
            y = torch.multiply(generated_samples[:, 0],cos)
            plt.plot(x,y , ".")
                
            plt.savefig(f'images_polarcircle_{epoch}.png')
            plt.close('all')
            
    





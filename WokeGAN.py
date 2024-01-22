# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:34:16 2024

@author: zacka
"""


import numpy as np
import math
import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import pickle
import sys
import pandas

torch.set_default_dtype(torch.float32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
cpu = 'cpu'


print('torch version:',torch.__version__)
print('device:', device)



with open('normalized_training_set.pkl', 'rb') as file:
    train_df = pickle.load(file)

with open('normalized_testing_set.pkl', 'rb') as file:
    test_df = pickle.load(file)

train_data = torch.Tensor(train_df.values).to(device)
train_conds = train_data[:,:10].to(device)
train_points = train_data[:,10:]
#print(train_points.size())

print(train_df['pid'].value_counts())
print(test_df['pid'].value_counts())
print(train_df.columns())
sys.exit()


#print(train_data)
#print(train_data.size())
train_size = train_data.size(dim = 0)


points = 3
conds = 10
noise_size = 6



# want to optimise these also add cuda possibilities
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
        nn.Linear(points*2 + conds, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
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
            nn.Linear(noise_size + conds, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, points*2),
            )

    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator().to(device)

# define hyperparameters

lr = 0.0005
num_epochs = 5000
loss_function = nn.BCELoss()





optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

counter = 0
loss_list = torch.Tensor([[counter, 0.5, 0.5]])

"""
# Load discriminator and generator models

discriminator.load_state_dict(torch.load('D_CGAN1.pth'))
generator.load_state_dict(torch.load('G_CGAN1.pth'))
"""

t=time.time()
for epoch in range(num_epochs):
    
        
    noise_vector = torch.randn(train_size, noise_size).to(device)
    latent_space_samples = torch.cat([train_conds, noise_vector], dim = 1)
    #print(latent_space_samples.size())
    
    
    #generate samples
    generated_samples = generator(latent_space_samples).to(device)
    generated_data = torch.cat([train_conds, generated_samples], dim = 1)
    #print(generated_data.size())
    
    generated_samples_labels = torch.zeros((train_size,1))
    real_samples_labels = torch.ones((train_size,1)).to(device)
    

    all_samples = torch.cat((train_data, generated_data)).to(device)
    all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels)).to(device)


    # Training the discriminator
    discriminator.zero_grad()
    output_discriminator = discriminator(all_samples).to(device)
    
    
    D_real = torch.mean(output_discriminator[:15])
    D_fake = torch.mean(output_discriminator[:-15])
    #calc loss and optimise
    

    loss_discriminator = loss_function(output_discriminator, all_samples_labels).to(device)
    loss_discriminator.backward()
    optimizer_discriminator.step()

    # Data for and training of the generator

    generator.zero_grad()
    noise_vector = torch.randn(train_size, noise_size).to(device)
    latent_space_samples = torch.cat([train_conds, noise_vector], dim = 1)
    #generate samples    
    generated_samples = generator(latent_space_samples).to(device)
    generated_data = torch.cat([train_conds, generated_samples], dim = 1)
    #output_discriminator_generated = discriminator(torch.cat((generated_samples, the_r),1))
    output_discriminator_generated = discriminator(generated_data).to(device)

    #calc loss and optimise
    loss_generator = loss_function(output_discriminator_generated, real_samples_labels).to(device)
    loss_generator.backward()
    optimizer_generator.step()
    
    
    # Create a list of tensors with counter, loss_g, and loss_d
    loss_g = loss_generator.detach()
    loss_d = loss_discriminator.detach()

    loss_entry = torch.tensor([[counter, loss_g, loss_d]])

    # Append the new entry to the list of losses
    loss_list = torch.cat((loss_list,loss_entry))
    print(loss_entry)

    counter += 1
    print("epoch: {} lossD: {} lossG: {}".format(epoch, loss_d, loss_g))
    # Show loss

    if epoch % 100 == 1 and epoch !=1:
        e = epoch
        print(f"Epoch: {e} loss D: {loss_discriminator}")
        print(f"Epoch: {e} loss G: {loss_generator}")
        t_e = 100*(time.time()-t)/epoch
        print("Epoch: {} time: {}".format(epoch,t_e))

        
        
        print(loss_list)
        loss_list = loss_list.to(cpu).numpy()
        plt.figure(figsize=(16,16))
        plt.subplot(2,2,1)
        #plt.title.set_text("Discriminator loss")
        plt.plot(loss_list[:,0], loss_list[:,1])
        plt.subplot(2,2,2)
        #plt.title.set_text("Generator loss")
        plt.plot(loss_list[:,0], loss_list[:,2])
        
        loss_list = torch.tensor(loss_list).to(device)
        
        noise_vector = torch.randn(train_size, noise_size).to(device)
        latent_space_samples = torch.cat([train_conds, noise_vector], dim = 1)
        #generate samples    
        generated_points = generator(latent_space_samples).to(device)
        
        
        
        points = generated_points.detach().to(cpu)
        points1 = np.array(points[600:610])
        xs1 = points1[:,:3]
        ys1 = points1[:,3:]
        points2 = np.array(train_points[600:610])
        xs2 = points2[:,:3]
        ys2 = points2[:,3:]
        
        print(xs1)
        print(ys1)
        plt.subplot(2,2,3)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        
        
        plt.plot(xs1, ys1, ".")
        
        
        plt.subplot(2,2,4)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        
        plt.plot(xs2, ys2, ".")
        plt.show()
        #plt.savefig(f'testset_{e}.png')
        plt.close('all')
        

"""

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
    
    

        """



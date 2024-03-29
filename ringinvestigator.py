
import sys
import numpy as np
import random
import math
import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import pickle
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import chi2

torch.set_default_dtype(torch.float32)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cpu= 'cpu'
print('torch version:',torch.__version__)
print('device:', device)

train_data = torch.load("centered_hits.pt").to(device)
train_size = train_data.size()[0]

with open("centered_scalar.pkl", 'rb') as file:
    scaling_dict = pickle.load(file)


radius_scalar = scaling_dict['radius']
max_hit = scaling_dict['max_hit']
max_center = scaling_dict['max_center']

points = 1
conds = 1

train_conds = train_data[:, 2].reshape(-1,1)
train_targets = train_data[:,:2]
train_data= train_data[:,:3]


class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
		nn.Linear(3, 256),
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

discriminator = Discriminator().to(device)

#added extra layer to generator, maybe needs it?
class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(3, 32),
			nn.ReLU(),
			nn.Linear(32, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
			)

	def forward(self, x):
		output = self.model(x)
		return output

generator = Generator().to(device)



# Load discriminator and generator models

discriminator.load_state_dict(torch.load('D_ringgan_batch.pth', map_location=torch.device('cpu')))
generator.load_state_dict(torch.load('G_ringgan_batch.pth', map_location=torch.device('cpu')))


f= 0.4
radii = torch.ones(39, 1)*(f-0.7)*3.3333


    

noise_vector = torch.randn(39, 2)
latent_space_samples = torch.cat([noise_vector, radii], dim = 1)
#generate samples    
generated_points = generator(latent_space_samples)
xs_gen = generated_points[:,0].detach().numpy()*max_hit
ys_gen = generated_points[:,1].detach().numpy()*max_hit

theta = np.linspace(0, 2*np.pi, 100)
radius = f*max_hit
x_circle = radius * np.cos(theta)
y_circle = radius * np.sin(theta)

# Plot the circle
fig, axes = plt.subplots(1, figsize=(8, 8))
axes.plot(x_circle, y_circle, color='red', linestyle='--', label='Circle')

# Plot your points xs and ys
# Assuming you have xs and ys defined somewhere
axes.plot(xs_gen, ys_gen, '.', label='Points')

axes.set_aspect('equal', 'box')
axes.legend()
plt.show()


range_arr = np.linspace(-3,3, num = 100)
noise_vector = []
radii = torch.ones(10000, 1)*(f-0.7)*10/3
for i in range_arr:
    for j in range_arr:
        noise_vector.append([i,j])
        
latent_space_samples = torch.cat([torch.Tensor(noise_vector), radii], dim = 1)
#generate samples    
generated_points = generator(latent_space_samples)
xs = generated_points[:,0].detach().numpy()
ys = generated_points[:,1].detach().numpy()


theta = np.arctan2(xs,ys).reshape(100,100)+np.pi
        
plt.imshow(theta, cmap='hot')#, interpolation='nearest')
plt.colorbar()  # Add colorbar to show the scale
plt.show()


x_re = train_data[:,0].detach().numpy().reshape(-1,1)*max_hit
y_re = train_data[:,1].detach().numpy().reshape(-1,1)*max_hit
r_re = radius_scalar.inverse_transform(train_data[:,2].detach().numpy().reshape(-1,1))

print(x_re.shape)
print(y_re.shape)
print(r_re.shape)

residual_real = np.sqrt(x_re**2 + y_re**2)-r_re

bin_num = 100

fig, axes = plt.subplots(1, figsize= (8,8))
n_re, bins_re, _ = axes.hist(residual_real,bins=bin_num)
plt.show()


noise_vector = torch.randn(train_size, 2)
latent_space_samples = torch.cat([torch.Tensor(noise_vector), train_conds], dim = 1)
#generate samples    
generated_points = generator(latent_space_samples)
xs = generated_points[:,0].detach().numpy()*max_hit
ys = generated_points[:,1].detach().numpy()*max_hit


x_gen = xs.reshape(-1,1)
y_gen = ys.reshape(-1,1)

print(x_re.shape)
print(y_re.shape)
print(r_re.shape)

residual_gen = np.sqrt(x_gen**2 + y_gen**2)-r_re

fig, axes = plt.subplots(1, figsize= (8,8))
n_gen, bins,_ = axes.hist(residual_gen, bins = bins_re)
plt.show()
#print(bins)
#print(n)



model_hist = np.zeros(1)
real_hist = np.zeros(1)

for i in range(bin_num):
    if n_gen[i] > 10 and n_re[i]>10:
        
        model_hist = np.append(model_hist,n_re[i])
        real_hist = np.append(real_hist,n_gen[i])



print('model:',model_hist)
print('real:', real_hist)
model_hist = model_hist[1:]
real_hist = real_hist[1:]

chi2_stat = np.sum(((model_hist-real_hist)**2)/(model_hist+real_hist))



pval = chi2.sf(chi2_stat, bin_num-1)

# Print results
print("Chi-squared Statistic:", chi2_stat)
print("pval:", pval)


fig, axes = plt.subplots(1, figsize=(8, 8))
axes.plot(x_circle, y_circle, color='red', linestyle='--', label='Circle')

# Plot your points xs and ys
# Assuming you have xs and ys defined somewhere
axes.plot(x_re[:39]*max_hit, y_re[:39]*max_hit, '.', label='Real')
axes.plot(xs_gen,ys_gen,'.', label="Gen")
#axes.set_aspect('equal', 'box')
axes.legend()
plt.show()


import sys
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

torch.set_default_dtype(torch.float32)
device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
cpu = 'cpu'

print('torch version:',torch.__version__)
print('device:', device)


train_data = torch.load("rnc_data.pt").to(device)
train_tensor = train_data
print(train_data)

with open("scaling_pions.pkl", 'rb') as file:
    scaling_dict = pickle.load(file)

r_scaler = scaling_dict["radius"]
conds_scaler = scaling_dict['conds']
#centers_scaler = scaling_dict[]

train_size = train_tensor.size()[0]


targets = 3
conds = 6
noise_size = targets

train_conds = train_tensor[:,:conds]
train_targets = train_tensor[:,conds:-1]
pid = train_tensor[:,-1]


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
        nn.Linear(targets + conds, 128),
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

discriminator = Discriminator().to(device)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(targets+conds, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, targets),
            )

    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator().to(device)

discriminator.load_state_dict(torch.load('D_rnc1.pth',map_location=torch.device('cpu')))
generator.load_state_dict(torch.load('G_rnc1.pth',map_location=torch.device('cpu')))


noise_vector = torch.randn(train_size, noise_size).to(device)
latent_space_samples = torch.cat([train_conds, noise_vector], dim = 1)
generated_points = generator(latent_space_samples).to(device)

# Assuming train_targets and generated_points are both tensors with 3 rows and N columns
train_targets_first_column = train_targets[:, 0]
generated_points_first_column = generated_points[:, 0]

# Calculate the difference
difference =  generated_points_first_column - train_targets_first_column

# Create a figure and two subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# Plot the first histogram (Radius Difference)
axes[0].hist(difference.cpu().detach().numpy(), bins=50)
axes[0].set_xlabel('Difference')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Radius Difference')

# Assuming train_targets and generated_points are both tensors with 3 rows and N columns
train_targets_second_row = train_targets[:,1]
generated_points_second_row = generated_points[:,1]
train_targets_third_row = train_targets[:,2]
generated_points_third_row = generated_points[:,2]

gen_cx = generated_points[:,1].detach().numpy()
gen_cy = generated_points[:,2].detach().numpy()

# Calculate the differences
difference_second_row = train_targets_second_row - generated_points_second_row
difference_third_row = train_targets_third_row - generated_points_third_row

# Add the differences in quadrature
squared_sum = difference_second_row**2 + difference_third_row**2
rooted_sum = torch.sqrt(squared_sum).detach().numpy()

# Plot the second histogram (Centres Difference)
axes[1].hist(rooted_sum, bins=50)
axes[1].set_xlabel('Rooted Sum of Differences')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Centres Difference')




threshold = 0.25

i_over_thresh = np.where(rooted_sum>threshold)[0]
i_under_thresh = np.where(rooted_sum<threshold)[0]

print(len(i_over_thresh))
print(len(i_under_thresh))
#

generated_array = generated_points.detach().numpy()
gen_over = generated_array[i_over_thresh]
gen_under = generated_array[i_under_thresh]
conds_array = train_conds.numpy()
conds_over = conds_array[i_over_thresh]
conds_under = conds_array[i_under_thresh]
targets_array = train_targets.numpy()
targets_over = targets_array[i_over_thresh]
targets_under = targets_array[i_under_thresh]
pid_array = pid.numpy()
pid_over = pid_array[i_over_thresh]
pid_under = pid_array[i_under_thresh]
print(np.unique(pid_over, return_counts=True))
print("========")
print(np.unique(pid_under, return_counts=True))


fig, axes = plt.subplots(2,2, figsize = (8,6))

axes[0,0].plot(targets_array[:,1],targets_array[:,2], '.')
axes[0,0].set_title("all centres true, thresh = {}".format(threshold))
axes[0,1].plot(targets_over[:,1],targets_over[:,2], '.')
axes[0,1].set_title("over thresh true")
axes[1,0].plot(targets_under[:,1],targets_under[:,2], '.')
axes[1,0].set_title('under thresh true')
plt.show()

fig, axes = plt.subplots(2,2, figsize = (8,6))

axes[0,0].plot(targets_over[:,1],targets_over[:,2], '.')
axes[0,0].set_title("over thresh true, thresh = {}".format(threshold))
axes[0,1].plot(targets_under[:,1],targets_under[:,2], '.')
axes[0,1].set_title("under thresh true")
axes[1,0].plot(gen_over[:,1],gen_over[:,2], '.')
axes[1,0].set_title('over thresh gen')
axes[1,1].plot(gen_under[:,1],gen_under[:,2], '.')
axes[1,1].set_title('under thresh gen')
plt.show()

fig, axes = plt.subplots(2,2, figsize = (8,6))

axes[0,0].hist(targets_over[:,0]-gen_over[:,0], bins = 50)
axes[0,0].set_title("radii over true")
axes[0,1].hist(targets_under[:,0]-gen_under[:,0], bins = 50)
axes[0,1].set_title("radii under true")
axes[1,0].hist(gen_over[:,0], bins = 50)
axes[1,0].set_title("radii over gen")
axes[1,1].hist(gen_under[:,0], bins = 50)
axes[1,1].set_title("radii under gen")
plt.show()
conds_name =('eta', 'phi', 'track_x','track_y', 'track_z', 'logp')
"""
for cond in range(conds):
    fig, axes = plt.subplots(2,1, figsize=(8,6))
    
    axes[0].hist(conds_over[:,cond], bins=75)
    axes[0].set_xlabel(conds_name[cond])
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Over Thresh')
    
    axes[1].hist(conds_under[:,cond], bins=75)
    axes[1].set_xlabel(conds_name[cond])
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Under Thresh')
    
    plt.show()
"""


sns.set_theme(style="white")


dall = pd.DataFrame(data=conds_array,
                 columns=conds_name)
dunder = pd.DataFrame(data=conds_under,
                 columns=conds_name)
dover = pd.DataFrame(data=conds_over,
                 columns=conds_name)


# Compute the correlation matrices
corr_all = dall.corr()
corr_under = dunder.corr()
corr_over = dover.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_all, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(3, figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_all, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax[0])
sns.heatmap(corr_under, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax[1])
sns.heatmap(corr_over, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax[2])


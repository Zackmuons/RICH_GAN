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
from scipy.stats import chi2_contingency
from scipy.stats import chisquare
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, RocCurveDisplay

torch.set_default_dtype(torch.float32)
device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
cpu = 'cpu'

print('torch version:',torch.__version__)
print('device:', device)


train_data = torch.load("rnc_data.pt").to(device)
train_tensor = train_data[:5000]
test_tensor = train_data[5000:]
print(train_data.size())




with open("scaling_pions.pkl", 'rb') as file:
    scaling_dict = pickle.load(file)

r_scaler = scaling_dict["radius"]
conds_scaler = scaling_dict['conds']

train_size = train_tensor.size()[0]
test_size = test_tensor.size()[0]




targets = 4
conds = 6
noise_size = targets

train_conds = train_tensor[:,:conds]
train_targets = train_tensor[:,conds:]


test_conds = test_tensor[:,:conds]
test_conds_fit = test_conds[:518]
test_conds_roc = test_conds[518:]
test_targets = test_tensor[:,conds:-1]
test_targets_fit = test_targets[:518]
test_targets_roc = test_targets[518:]

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

# define hyperparameters

lr = 5E-5
num_epochs = int(2E6)
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

counter = 0
loss_list = torch.Tensor([[counter, 0.7, 0.7]])

plot_num = int(5E4)
test_num = int(1E2)

# Load discriminator and generator models
"""
discriminator.load_state_dict(torch.load('D_rnc2.pth',map_location=torch.device('cpu')))
generator.load_state_dict(torch.load('G_rnc2.pth', map_location=torch.device('cpu')))
"""

AUC_vals = torch.tensor([1,0]).unsqueeze(0)

t=time.time()
for epoch in range(num_epochs):
    
        
    noise_vector = torch.randn(train_size, noise_size).to(device)
    latent_space_samples = torch.cat([train_conds, noise_vector], dim = 1)
    #print(latent_space_samples.size())
    
    
    #generate samples
    generated_samples = generator(latent_space_samples).to(device)
    generated_data = torch.cat([train_conds, generated_samples], dim = 1)
    #print(generated_data.size())
    
    
    generated_samples_labels = torch.zeros((train_size,1)).to(device)
    real_samples_labels = torch.ones((train_size,1)).to(device)
    
    train_tensor = torch.cat([train_conds, train_targets], axis =1)
    #print(train_tensor.size())
    #print(generated_data.size())
    all_samples = torch.cat((train_tensor, generated_data)).to(device)
    all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels)).to(device)


    # Training the discriminator
    discriminator.zero_grad()
    output_discriminator = discriminator(all_samples).to(device)
    
    #print(output_discriminator.max())
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
    #print(loss_entry)

    counter += 1
    #print("epoch: {} lossD: {} lossG: {}".format(epoch, loss_d, loss_g))
    # Show loss

    
    if (epoch % test_num == 0 and epoch !=0) or epoch == num_epochs-1:
        e = epoch
        print("Epoch: {} loss D: {}".format(e, loss_discriminator))
        print("Epoch: {} loss G: {}".format(e, loss_generator))
        t_e = 100*(time.time()-t)/epoch
        print("Epoch: {} time: {}".format(epoch,t_e))
        
        noise_vector = torch.randn(train_size, noise_size).to(device)
        latent_space_samples = torch.cat([train_conds, noise_vector], dim = 1)
        generated_points = generator(latent_space_samples).to(device)
        
        # Assuming train_targets and generated_points are both tensors with 3 rows and N columns
        train_targets_first_column = train_targets[:, 0]
        generated_points_first_column = generated_points[:, 0]
        
        # Calculate the difference
        difference =  generated_points_first_column - train_targets_first_column
        
        if (epoch % plot_num ==0) or (epoch == 100):
            # Create a figure and two subplots
            fig, axes = plt.subplots(3, 2, figsize=(8, 6))
            
            # Plot the first histogram (Radius Difference)
            axes[0,0].hist(difference.cpu().detach().numpy(), bins=50)
            axes[0,0].set_xlabel('Difference')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].set_title('Radius Difference')
            
            # Assuming train_targets and generated_points are both tensors with 3 rows and N columns
            train_targets_second_row = train_targets[:,1]
            generated_points_second_row = generated_points[:,1]
            train_targets_third_row = train_targets[:,2]
            generated_points_third_row = generated_points[:,2]
            n_hits_real = train_targets[:,3]
            n_hits_gen = generated_points[:,3]
            
            # Calculate the differences
            difference_second_row = train_targets_second_row - generated_points_second_row
            difference_third_row = train_targets_third_row - generated_points_third_row
            
            # Add the differences in quadrature
            squared_sum = difference_second_row**2 + difference_third_row**2
            rooted_sum = torch.sqrt(squared_sum)
            
            # Plot the second histogram (Centres Difference)
            axes[0,1].hist(rooted_sum.cpu().detach().numpy(), bins=50)
            axes[0,1].set_xlabel('Rooted Sum of Differences')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].set_title('Centres Difference')
            
            n_hits_diff = n_hits_real - n_hits_gen
            
            axes[2,0].hist(n_hits_diff.cpu().detach().numpy(), bins = 50)
            axes[2,0].set_title("n-hits_diff")
            
            
            #print(loss_list)
            loss_plot = loss_list.to(cpu).numpy()
            # plt.figure(figsize=(16,16))
            # plt.subplot(2,2,1)
            # plt.title.set_text("Discriminator loss")
            # plt.plot(loss_list[:,0], loss_plot[:,1])
            # plt.subplot(2,2,2)
            # plt.title.set_text("Generator loss")
            # plt.plot(loss_list[:,0], loss_plot[:,2])
            
            axes[1,0].plot(loss_list[:,0], loss_plot[:,1])
            axes[1,0].set_title('Discriminator loss')
            axes[1,0].set_ylabel('Loss')
            axes[1,0].set_xlabel('Epoch')
            
            axes[1,1].plot(loss_list[:,0], loss_plot[:,2])
            axes[1,1].set_title('Generator loss')
            axes[1,1].set_ylabel('Loss')
            axes[1,1].set_xlabel('Epoch')
            # Adjust layout
            plt.tight_layout()
            
            # Show the plot
            plt.savefig(f"test_set_{epoch}.png")
            #plt.show()
        
        clf = GradientBoostingClassifier()
        
        noise_vector = torch.randn(test_size, noise_size).to(device)
        latent_space_samples = torch.cat([test_conds, noise_vector], dim = 1)
        generated_points = generator(latent_space_samples).to(cpu).detach().numpy()
        
        gen_fit = generated_points[:518,:-1]
        gen_roc = generated_points[518:,:-1]
        
        X_gen_fit = np.concatenate([test_conds_fit.to(cpu), gen_fit], axis = 1)
        X_gen_roc = np.concatenate([test_conds_roc.to(cpu), gen_roc], axis = 1)
        X_real_fit = np.concatenate([test_conds_fit.to(cpu), test_targets_fit.to(cpu)], axis =1)
        X_real_roc = np.concatenate([test_conds_roc.to(cpu), test_targets_roc.to(cpu)], axis = 1)
        # Label the data
        labels_generated = np.zeros(X_gen_fit.shape[0])  # Label 0 for generated data
        labels_real = np.ones(X_real_fit.shape[0])  # Label 1 for real data

        # Combine real and generated data
        X_fit = np.vstack((X_real_fit, X_gen_fit))
        X_roc = np.vstack((X_real_roc, X_gen_roc))
        y_fit = np.concatenate((labels_real, labels_generated))
        y_roc = np.concatenate((labels_real, labels_generated))
        

        # Shuffle the data
        indices = np.random.permutation(X_fit.shape[0])
        X_fit = X_fit[indices]
        y_fit = y_fit[indices]

        # Train the classifier
        clf.fit(X_fit, y_fit)

        # Predict probabilities for positive class (real data)
        y_pred = clf.predict_proba(X_roc)[:, 1]

        # Calculate ROC AUC score
        roc_auc = roc_auc_score(y_roc, y_pred)
        print(f"Epoch {epoch}: ROC AUC Score = {roc_auc}")
        AUC_vals = torch.cat([AUC_vals, torch.tensor([roc_auc, epoch]).unsqueeze(0)], axis = 0)
        
        # Plot ROC curve
        if (epoch % plot_num ==0) or (epoch == 100):
            plt.figure()
            roc_display = RocCurveDisplay.from_estimator(clf, X_roc, y_roc)
            plt.title(f"ROC Curve - Epoch {epoch}")
            plt.savefig(f"ROC_curve_{epoch}")
            
            plt.figure()
            plt.plot(AUC_vals[:,1], AUC_vals[:,0])
            plt.title(f"AUC-EPOCH {epoch}")
            plt.savefig(f"AUC_EPOCH_{epoch}.png")
            
            
            gan_info = {
                    'discriminator_state_dict': discriminator.state_dict(),
                    'generator_state_dict': generator.state_dict(),
                    'AUC_vals': AUC_vals
                    }
            auc_2dp = round(roc_auc,2)
            
            torch.save(gan_info, f'rnc_{epoch}_roc_{auc_2dp}.pth')
            
# Create a dictionary to save GAN information



# Save discriminator and generator models




torch.save(discriminator.state_dict(), 'D_rnc2.pth')
torch.save(generator.state_dict(), 'G_rnc2.pth')
# Save GAN information

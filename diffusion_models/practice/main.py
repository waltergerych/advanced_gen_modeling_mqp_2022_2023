# Code from https://github.com/azad-academy/denoising-diffusion-model/blob/main/diffusion_model_demo.ipynb

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_checkerboard,make_circles,make_moons,make_s_curve,make_swiss_roll
from helper_plot import hdr_plot_style
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils import * 

from model import ConditionalModel
from ema import EMA
from evaluate import *
from classifier import *
from diffusion import *
from gan import *

hdr_plot_style()
swiss_roll, _ = make_swiss_roll(10**4,noise=0.1)
swiss_roll = swiss_roll[:, [0, 2]]/10.0

s_curve, _= make_s_curve(10**4, noise=0.1)
s_curve = s_curve[:, [0, 2]]/10.0

moons, _ = make_moons(10**4, noise=0.1)

data = s_curve.T
dataset = torch.Tensor(data.T).float()


# fig,axes = plt.subplots(1,3,figsize=(20,5))

# axes[0].scatter(*data, alpha=0.5, color='white', edgecolor='gray', s=5)
# axes[0].axis('off')

# data = swiss_roll.T
# axes[1].scatter(*data, alpha=0.5, color='white', edgecolor='gray', s=5)
# axes[1].axis('off')
# # dataset = torch.Tensor(data.T).float()

# data = moons.T
# axes[2].scatter(*data, alpha=0.5, color='white', edgecolor='gray', s=3)
# axes[2].axis('off')
# # dataset = torch.Tensor(data.T).float()
# plt.show()

# load the datasets
train_x, train_y = load_data('../../dataset/UCI_HAR_Dataset', 'train')
test_x, test_y = load_data('../../dataset/UCI_HAR_Dataset', 'test')

classes = ['WALKING', 'U-STAIRS', 'D-STAIRS', 'SITTING', 'STANDING', 'LAYING']

labels = train_y
dataset = train_x

# Define the number of features from the dataset to use. Must be 561 or less
num_features = 40
num_datapoints = 10

# Select only the first x features
dataset = dataset[:, :num_features]

# Number of time steps
NUM_STEPS = 100
NUM_REVERSE_STEPS = 10000
# Number of graphs to plot to show the addition of noise over time (not including X_0)
NUM_DIVS = 10

# Normal diffusion for dataset
# diffusion = forward_diffusion(dataset, NUM_STEPS, plot=False)
# model = reverse_diffusion(dataset, diffusion, NUM_REVERSE_STEPS, plot=False)
# torch.save(model.state_dict(), f'./models/test_model.pth')

# Makes diffusion model for each class for the Classifier
# models = []
# diffusions = []

# original_data, original_labels = dataset, labels

# for i in range(len(classes)):
#     dataset, labels = get_activity_data(original_data, original_labels, i)

#     diffusion = forward_diffusion(dataset, NUM_STEPS, plot=False)
#     print("Starting training for class " + str(classes[i]))
#     model = reverse_diffusion(dataset, diffusion, NUM_REVERSE_STEPS, plot=False)
#     models.append(model)
#     diffusions.append(diffusion)

#     torch.save(model.state_dict(), f'./models/model_40_{i}.pth')

# model = ConditionalModel(NUM_STEPS, dataset.size(1))
# model.load_state_dict(torch.load('./models/har_diffusion.pth'))

##################
### EVALUATION ###
##################

labels = test_y
dataset = test_x
dataset = dataset[:, :num_features]
input_size = dataset.shape[1]
num_to_gen = 150
test_train_ratio = .3

generator_input_size = 128
hidden_size = 512

# Get real data set to be the same size as generated data set
dataset = dataset[:num_to_gen*len(classes)]
labels = labels[:num_to_gen*len(classes)]

# Get data for each class
diffusion_data = []
diffusion_labels = []
gan_data = []
gan_labels = []

# Get denoising variables
ddpm = get_denoising_variables(NUM_STEPS)

for i in range(len(classes)):
    # Load trained diffusion model
    model = ConditionalModel(NUM_STEPS, dataset.size(1))
    model.load_state_dict(torch.load(f'./models/model_40_{i}.pth'))

    # Load trained GAN model
    generator = Generator(generator_input_size, hidden_size, dataset.size(1))
    generator.load_state_dict(torch.load(f'./gan/generator/G_{classes[i]}.pth'))

    # Get outputs of both models
    diffusion_output = get_model_output(model, input_size, ddpm, NUM_STEPS, num_to_gen)
    gan_output = generate_data([generator], num_to_gen, generator_input_size)
    
    # CODE TO GRAPH 10 PLOTS OF REMOVING NOISE FOR EACH CLASS 
    # --> MUST CHANGE 'get_model_output' to return x_seq rather than x_seq[-1]
    # true_batch, true_labels = get_activity_data(dataset, labels, i)
    # for j in range(0, NUM_STEPS, 10):
    #     perform_pca(true_batch, output[j], f'T{100-j}')
    # perform_pca(true_batch, output[-1], 'T0')
    # plt.show()

    # Add model outputs and labels to lists
    diffusion_data.append(diffusion_output)
    diffusion_labels.append(torch.mul(torch.ones(num_to_gen), i))
    gan_data.append(gan_output[0])
    gan_labels.append(torch.mul(torch.ones(num_to_gen), i))
    print("Generated data for " + str(classes[i]))

# Concatenate data into single tensor
diffusion_data, diffusion_labels = torch.cat(diffusion_data), torch.cat(diffusion_labels)
gan_data, gan_labels = torch.cat(gan_data), torch.cat(gan_labels)

# Do PCA analysis for fake/real and subclasses
pca_with_classes(dataset, labels, diffusion_data, diffusion_labels, classes, overlay_heatmap=False)

# Show PCA for each class
for i in range(len(classes)):
    true_batch, true_labels = get_activity_data(dataset, labels, i)
    fake_batch, fake_labels = get_activity_data(diffusion_data, diffusion_labels, i)
    perform_pca(true_batch, fake_batch, f'{classes[i]}')
plt.show()

# Machine evaluation for diffusion and GAN data
print('Testing data from diffusion model')
binary_machine_evaluation(dataset, labels, diffusion_data, diffusion_labels, classes, test_train_ratio)
multiclass_machine_evaluation(dataset, labels, diffusion_data, diffusion_labels, test_train_ratio)
separability(dataset, diffusion_data, test_train_ratio)

print('Testing data from gan model')
binary_machine_evaluation(dataset, labels, gan_data, gan_labels, classes, test_train_ratio)
multiclass_machine_evaluation(dataset, labels, gan_data, gan_labels, test_train_ratio)
separability(dataset, gan_data, test_train_ratio)


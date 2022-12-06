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
num_to_gen = 500

# Get data for each class
generated_data_x = []
generated_data_y = []

# Get denoising variables
ddpm = get_denoising_variables(NUM_STEPS)

for i in range(len(classes)):
    # Load trained model
    model = ConditionalModel(NUM_STEPS, dataset.size(1))
    model.load_state_dict(torch.load(f'./models/model_40_{i}.pth'))

    # Get output
    output = get_model_output(model, input_size, ddpm, NUM_STEPS, num_to_gen)
    
    # CODE TO GRAPH 10 PLOTS OF REMOVING NOISE FOR EACH CLASS 
    # --> MUST CHANGE 'get_model_output' to return x_seq rather than x_seq[-1]
    # true_batch, true_labels = get_activity_data(dataset, labels, i)
    # for j in range(0, NUM_STEPS, 10):
    #     perform_pca(true_batch, output[j], f'T{100-j}')
    # perform_pca(true_batch, output[-1], 'T0')
    # plt.show()

    generated_data_x.append(output)
    generated_data_y.append(torch.mul(torch.ones(num_to_gen), i))
    print("Generated data for " + str(classes[i]))

gen_x, gen_y = torch.cat(generated_data_x), torch.cat(generated_data_y)

# Do PCA analysis for fake/real and subclasses
perform_pca_har_classes(dataset, labels, classes)
perform_pca_har_classes(gen_x, gen_y, classes)
pca_class_real_and_fake(dataset, labels, gen_x, gen_y, classes)

# Show PCA for each class
for i in range(len(classes)):
    true_batch, true_labels = get_activity_data(dataset, labels, i)
    fake_batch, fake_labels = get_activity_data(gen_x, gen_y, i)
    perform_pca(true_batch, fake_batch, f'PCA for class {classes[i]}')
plt.show()

# Split into testing and training for classifier
real_train_x, real_test_x, real_train_y, real_test_y = train_test_split(dataset, labels, test_size=.3)
fake_train_x, fake_test_x, fake_train_y, fake_test_y = train_test_split(gen_x, gen_y, test_size=.3)

for i in range(len(classes)):
    print(f'\nEvaluating class {classes[i]}')

    # Train classifier on real data
    print('Testing classifier trained on real data')
    classifier = build_binary_classifier(real_train_x, real_train_y, classes, i)
    
    print('Evaluating on real data')
    test_binary_classifier(classifier, real_test_x, real_test_y, classes, i)

    print('Evaluating on fake data')
    test_binary_classifier(classifier, fake_test_x, fake_test_y, classes, i)

    # Train classifier on fake data
    print('Testing classifier trained on fake data')
    classifier = build_binary_classifier(fake_train_x, fake_train_y, classes, i)

    print('Evaluating on real data')
    test_binary_classifier(classifier, real_test_x, real_test_y, classes, i)

    print('Evaluating on fake data')
    test_binary_classifier(classifier, fake_test_x, fake_test_y, classes, i)


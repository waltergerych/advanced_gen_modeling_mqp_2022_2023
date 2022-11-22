# Code from https://github.com/azad-academy/denoising-diffusion-model/blob/main/diffusion_model_demo.ipynb

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_checkerboard,make_circles,make_moons,make_s_curve,make_swiss_roll
from helper_plot import hdr_plot_style
import torch
import torch.optim as optim
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
num_features = 561

# Select only the first x features
# First 80 features are all acceleration
dataset = dataset[:, :num_features]

# Number of time steps
NUM_STEPS = 100
NUM_REVERSE_STEPS = 1000
# Number of graphs to plot to show the addition of noise over time (not including X_0)
NUM_DIVS = 10

models = []
diffusions = []

original_data, original_labels = dataset, labels

# for i in range(len(classes)):
#     dataset, labels = get_activity_data(original_data, original_labels, i)

#     diffusion = forward_diffusion(dataset, NUM_STEPS, plot=False)
#     print("Starting training for class " + str(classes[i]))
#     model = reverse_diffusion(dataset, diffusion, NUM_REVERSE_STEPS, plot=False)
#     models.append(model)
#     diffusions.append(diffusion)

#     torch.save(model.state_dict(), f'./models/model_561_{i}.pth')

# model = ConditionalModel(NUM_STEPS, dataset.size()[1])
# model.load_state_dict(torch.load('./models/har_diffusion.pth'))


### Evaluation ###

print("Starting evaluation")

labels = test_y
dataset = test_x
dataset = dataset[:, :num_features]

# model = ConditionalModel(NUM_STEPS, dataset.size()[1])
# model.load_state_dict(torch.load('./models/har_diffusion_walking.pth'))

# output = get_model_output(model, dataset, diffusion.alphas_bar_sqrt, diffusion.one_minus_alphas_bar_sqrt, NUM_STEPS)
# print(output)
# perform_pca(dataset, output)
# graph_two_features(dataset, output)


# Evaluate on a trained classifier
generated_data_x = []
generated_data_y = []
for i in range(len(classes)):
    model = ConditionalModel(NUM_STEPS, dataset.size()[1])
    model.load_state_dict(torch.load(f'./models/model_561_{i}.pth'))
    diffusion = forward_diffusion(dataset, NUM_STEPS, plot=False)
    output = get_model_output(model, dataset, diffusion.alphas_bar_sqrt, diffusion.one_minus_alphas_bar_sqrt, NUM_STEPS)
    generated_data_x.append(output)
    generated_data_y.append(torch.mul(torch.ones(output.size()[0]), i))
    print("Generated data for " + str(classes[i]))

gen_x = generated_data_x[0]
for tensor in generated_data_x[1:]:
    gen_x = torch.cat((gen_x, tensor), 0)

generated_x, generated_y = gen_x, torch.cat(generated_data_y)

input_size = 128
classifier_path = './classifiers/real_trained_classifier.pth'

classifier = Classifier(num_features, input_size)
classifier.load_state_dict(torch.load(classifier_path))

real_accuracy = get_accuracy(classifier, dataset, labels, class_stats=True)
fake_accuracy = get_accuracy(classifier, generated_x, generated_y, class_stats=True)


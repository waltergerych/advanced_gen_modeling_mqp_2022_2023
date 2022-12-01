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
num_features = 40
num_datapoints = 10

# Select only the first x features
dataset = dataset[:, :num_features]

# Number of time steps
NUM_STEPS = 100
NUM_REVERSE_STEPS = 10000
# Number of graphs to plot to show the addition of noise over time (not including X_0)
NUM_DIVS = 10

# cat = [(0,1), (0,1), (1,1), (0,1)]
# dataset = torch.tensor(cat, dtype=torch.float32)

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

### Evaluation ###

# print("Starting evaluation")

labels = test_y
dataset = test_x
dataset = dataset[:, :num_features]

# model = ConditionalModel(NUM_STEPS, dataset.size(1))
# model.load_state_dict(torch.load('./models/test_model.pth'))

# # cat = [(1,1), (1,1), (1,1), (0,1), (1,1), (0,1)]
# # dataset = torch.tensor(cat, dtype=torch.float32)

# diffusion = Diffusion(NUM_STEPS)
# # noise = q_x_cat(dataset, torch.tensor([NUM_STEPS-1]), diffusion, torch.tensor([2]))
# noise = q_x(dataset, torch.tensor([NUM_STEPS-1]), diffusion)

# graph_two_features(dataset, noise)
# for i in range(1, NUM_DIVS + 1):
#     # output = get_model_output(model, dataset, diffusion.alphas_bar_sqrt, diffusion.one_minus_alphas_bar_sqrt, int(i * (NUM_STEPS/NUM_DIVS)))
#     output = use_model(model, dataset, int(i * (NUM_STEPS/NUM_DIVS) - 1), num_datapoints)
#     graph_two_features(dataset, output)
# plt.show()

# output = use_model(model, dataset, NUM_STEPS-1, 10)
# graph_two_features(dataset, output)

# output = get_model_output(model, dataset, diffusion.alphas_bar_sqrt, diffusion.one_minus_alphas_bar_sqrt, NUM_STEPS)
# perform_pca(dataset, output)
# graph_two_features(dataset, output, noise)


# Evaluate on a trained classifier
generated_data_x = []
generated_data_y = []
for i in range(len(classes)):
    model = ConditionalModel(NUM_STEPS, dataset.size(1))
    model.load_state_dict(torch.load(f'./models/model_40_{i}.pth'))
    batch_data, batch_labels = get_activity_data(dataset, labels, i)
    diffusion = forward_diffusion(batch_data, NUM_STEPS, plot=False)
    output = get_model_output(model, batch_data, diffusion.alphas_bar_sqrt, diffusion.one_minus_alphas_bar_sqrt, NUM_STEPS)
    generated_data_x.append(output)
    generated_data_y.append(torch.mul(torch.ones(output.size(0)), i))
    print("Generated data for " + str(classes[i]))

gen_x, gen_y = torch.cat(generated_data_x), torch.cat(generated_data_y)

for i in range(len(classes)):
    print(f'\nEvaluating class {classes[i]}')
    # Train classifier on real data or load from previously trained and saved model
    classifier = build_binary_classifier(dataset, labels, classes, i)
    # path = './classifiers/rf_{i}.joblib'
    # classifier = load_binary_classifier(path)
    print('Testing classifier trained on real data')
    # Evaluate on real data for class i
    print('Evaluating on real data')
    test_binary_classifier(classifier, dataset, labels, classes, i)
    # Evaluate on fake data for class i
    print('Evaluating on fake data')
    test_binary_classifier(classifier, gen_x, gen_y, classes, i)

    # Train classifier on real data or load from previously trained and saved model
    classifier = build_binary_classifier(gen_x, gen_y, classes, i)
    print('Testing classifier trained on fake data')
    # Evaluate on real data for class i
    print('Evaluating on real data')
    test_binary_classifier(classifier, dataset, labels, classes, i)
    # Evaluate on fake data for class i
    print('Evaluating on fake data')
    test_binary_classifier(classifier, gen_x, gen_y, classes, i)


    # Train classifier on fake data or load from previously trained and saved model
    # classifier = build_binary_classifier(gen_x, gen_y, classes, i)
    # # path = './classifiers/rf_{i}.joblib'
    # # classifier = load_binary_classifier(path)

    # # Evaluate on real data for class i
    # test_binary_classifier(classifier, dataset, labels, classes, i)
    # # Evaluate on fake data for class i
    # test_binary_classifier(classifier, gen_x, gen_y, classes, i)


# input_size = 128
# classifier_path = './classifiers/real_trained_classifier.pth'

# classifier = Classifier(num_features, input_size)
# classifier.load_state_dict(torch.load(classifier_path))

# real_accuracy = get_accuracy(classifier, dataset, labels, class_stats=True)
# fake_accuracy = get_accuracy(classifier, generated_x, generated_y, class_stats=True)


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
# train_x, train_y = load_data('../../dataset/UCI_HAR_Dataset', 'train')
# test_x, test_y = load_data('../../dataset/UCI_HAR_Dataset', 'test')

# classes = ['WALKING', 'U-STAIRS', 'D-STAIRS', 'SITTING', 'STANDING', 'LAYING']

# labels = train_y
# dataset = train_x

# # Define the number of features from the dataset to use. Must be 561 or less
# num_features = 2

# # Select only the first x features
# # First 80 features are all acceleration
# dataset = dataset[:, :num_features]

# dataset, labels = get_activity_data(dataset, labels, 0)

# Number of time steps
NUM_STEPS = 100
# Number of graphs to plot to show the addition of noise over time (not including X_0)
NUM_DIVS = 10
betas = make_beta_schedule(schedule='sigmoid', n_timesteps=NUM_STEPS, start=1e-5, end=0.5e-2)

alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

# Add t time steps of noise to the data x
def q_x(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
    return (alphas_t * x_0 + alphas_1_m_t * noise)


def visualize_forward(dataset, num_steps, num_divs):
    fig, axs = plt.subplots(1, num_divs + 1, figsize=(28, 3))
    axs[0].scatter(dataset[:, 0], dataset[:, 1],color='white',edgecolor='gray', s=5)
    axs[0].set_axis_off()
    axs[0].set_title('$q(\mathbf{x}_{'+str(0)+'})$')
    for i in range(1, num_divs + 1):
        q_i = q_x(dataset, torch.tensor([i * int(num_steps/num_divs) - 1]))
        axs[i].scatter(q_i[:, 0], q_i[:, 1],color='white',edgecolor='gray', s=5)
        axs[i].set_axis_off()
        axs[i].set_title('$q(\mathbf{x}_{'+str(i*int(num_steps/num_divs))+'})$')
    plt.show()

def visualize_backward(model, dataset, num_steps, num_divs, alphas, betas, one_minus_alphas_bar_sqrt, reverse=False):
    x_seq = p_sample_loop(model, dataset.shape,num_steps,alphas,betas,one_minus_alphas_bar_sqrt)
    fig, axs = plt.subplots(1, num_divs+1, figsize=(28, 3))
    for i in range(num_divs + 1):
        cur_x = x_seq[i * int(num_steps/num_divs)].detach()
        axs[i if not reverse else num_divs-i].scatter(cur_x[:, 0], cur_x[:, 1],color='white',edgecolor='gray', s=5)
        axs[i if not reverse else num_divs-i].set_axis_off()
        axs[i if not reverse else num_divs-i].set_title('$q(\mathbf{x}_{'+str(int((num_divs-i)*(num_steps)/num_divs))+'})$')

# Visualize the forward process
visualize_forward(dataset, NUM_STEPS, NUM_DIVS)

# Used to calculate the mean and variance of the data.  Can be used to assess how close the data is to
# the normal distribution (all noise) 
posterior_mean_coef_1 = (betas * torch.sqrt(alphas_prod_p) / (1 - alphas_prod))
posterior_mean_coef_2 = ((1 - alphas_prod_p) * torch.sqrt(alphas) / (1 - alphas_prod))
posterior_variance = betas * (1 - alphas_prod_p) / (1 - alphas_prod)
posterior_log_variance_clipped = torch.log(torch.cat((posterior_variance[1].view(1, 1), posterior_variance[1:].view(-1, 1)), 0)).view(-1)

def q_posterior_mean_variance(x_0, x_t, t):
    coef_1 = extract(posterior_mean_coef_1, t, x_0)
    coef_2 = extract(posterior_mean_coef_2, t, x_0)
    mean = coef_1 * x_0 + coef_2 * x_t
    var = extract(posterior_log_variance_clipped, t, x_0)
    return mean, var


### TRAINING ###

print("Starting training")

model = ConditionalModel(NUM_STEPS, dataset.size()[1])
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Create EMA model
ema = EMA(0.9)
ema.register(model)

batch_size = 128
for t in range(NUM_STEPS):
    # X is a torch Variable
    permutation = torch.randperm(dataset.size()[0])
    for i in range(0, dataset.size()[0], batch_size):
        # Retrieve current batch
        indices = permutation[i:i+batch_size]
        batch_x = dataset[indices]
        # Compute the loss
        loss = noise_estimation_loss(model, batch_x,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,NUM_STEPS)
        # Before the backward pass, zero all of the network gradients
        optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Perform gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        # Calling the step function to update the parameters
        optimizer.step()
        # Update the exponential moving average
        ema.update(model)
    # Print loss
    if (t % int(NUM_STEPS/NUM_DIVS) == 0):
        print(f'{t}:\t{loss}')
        visualize_backward(model, dataset, NUM_STEPS, NUM_DIVS, alphas, betas, one_minus_alphas_bar_sqrt, reverse=True)

plt.show()

torch.save(model.state_dict(), './models/har_diffusion_walking.pth')


### Evaluation ###

print("Starting evaluation")

# labels = test_y
# dataset = test_x
# dataset = dataset[:, :num_features]
# dataset, labels = get_activity_data(dataset, labels, 0)

model = ConditionalModel(NUM_STEPS, dataset.size()[1])
model.load_state_dict(torch.load('./models/har_diffusion_walking.pth'))

output = get_model_output(model, dataset, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, NUM_STEPS)
print(output)
# perform_pca(dataset, output)
graph_two_features(dataset, output)


# Evaluate on a trained classifier
# fake, real = output, dataset
# input_size = 128
# classifier_path = './classifiers/real_trained_classifier.pth'

# classifier = Classifier(len(real[0][0]), input_size)
# classifier.load_state_dict(torch.load(classifier_path))

# real_accuracy = get_accuracy(classifier, real, labels, class_stats=True)
# fake_accuracy = get_accuracy(classifier, fake, labels, class_stats=True)


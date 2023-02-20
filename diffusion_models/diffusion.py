# Code from https://github.com/azad-academy/denoising-diffusion-model/blob/main/diffusion_model_demo.ipynb

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_checkerboard,make_circles,make_moons,make_s_curve,make_swiss_roll
from helper_plot import hdr_plot_style
import torch
import torch.optim as optim
from utils import *
from sklearn.preprocessing import OneHotEncoder

from model import ConditionalModel
from model import ConditionalMultinomialModel
from ema import EMA
from evaluate import *
from classifier import *
import seaborn as sns

class Diffusion():
    def __init__(self, num_steps):
        """
        Class constructor for feed-forward model.

        All fields of model are torch.Tensor's of size(num_steps)

        Args:
            num_steps (int): number of time steps desired in diffusion model
        """
        super().__init__()
        self.num_steps = num_steps
        # For continuous noise
        # Beta scheduler
        self.betas = make_beta_schedule(schedule='linear', n_timesteps=num_steps, start=1e-5, end=0.5e-2)
        # Alphas
        self.alphas = 1 - self.betas
        # Cumulative product of alphas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        # Square root of the cumulative product of alphas
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        # One minus the cumulative product of alphas
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        # For categorical noise
        # Log of alphas
        self.log_alphas = np.log(self.alphas)
        # One minus log of alphas
        self.one_minus_log_alphas = log_1_min_a(self.log_alphas)
        # Cumulative sum of log of alphas
        self.log_cumprod_alpha = np.cumsum(self.log_alphas)
        # One minus log of cumulative sum
        self.log_1_min_cumprod_alpha = log_1_min_a(self.log_cumprod_alpha)

def get_denoising_variables(num_steps):
    """Calculates the variables used in the denoising process and captures them in the class 'Diffusion"

    Args:
        num_steps (int): the number of steps in the forward diffusion

    Returns:
        diffusion (Diffusion): a class encapsulating the denoising variables
    """
    diffusion = Diffusion(num_steps)

    return diffusion

# Add t time steps of noise to the data x
def q_x(x_0, t, model, noise=None):
    """Function to add t time steps of noise to continuous data x

    Args:
        x_0 (torch.Tensor): the data to add noise to
        t (torch.Tensor): the number of time steps to add
        model (class: Diffusion): a diffusion model class encapsulating proper constants for forward diffusion
                                Constants calculated from num_steps input to class constructor

    Returns:
        (torch.Tensor): the data with the noise added to it
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(model.alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(model.one_minus_alphas_bar_sqrt, t, x_0)
    return (alphas_t * x_0 + alphas_1_m_t * noise)

def visualize_forward(dataset, num_steps, num_divs, diffusion):
    """Vizualizes the forward diffusion process

    Args:
        dataset (torch.Tensor): the original dataset without noise
        num_steps (int): number of steps of noise to be removed
        num_divs (int): number of graphs to plot
        diffusion (class: Diffusion): a diffusion class which captures forward diffusion variables
    """
    fig, axs = plt.subplots(1, num_divs + 1, figsize=(28, 3))
    axs[0].scatter(dataset[:, 0], dataset[:, 1],color='white',edgecolor='gray', s=5)
    axs[0].set_axis_off()
    axs[0].set_title('$q(\mathbf{x}_{'+str(0)+'})$')
    for i in range(1, num_divs + 1):
        # q_i = q_x(dataset, torch.tensor([i * int(num_steps/num_divs) - 1]), diffusion)
        q_i = q_x_cat(dataset, torch.tensor([i * int(num_steps/num_divs) - 1]), diffusion, torch.tensor([2]))
        axs[i].scatter(q_i[:, 0], q_i[:, 1],color='white',edgecolor='gray', s=5)
        axs[i].set_axis_off()
        axs[i].set_title('$q(\mathbf{x}_{'+str(i*int(num_steps/num_divs))+'})$')
    plt.show()

def visualize_backward(model, dataset, num_steps, num_divs, diffusion, heatmap=False, reverse=False):
    """Vizualizes the backwards diffusion process

    Args:
        model (class: ConditionalModel): the model being used
        dataset (torch.Tensor): the original dataset without noise
        num_steps (int): number of steps of noise to be removed
        num_divs (int): number of graphs to plot
        diffusion (class: Diffusion): a diffusion class which captures forward diffusion variables
        reverse (bool): If true, will plot the graphs in reverse
    """
    x_seq = p_sample_loop(model, dataset.shape,num_steps,diffusion.alphas,diffusion.betas,diffusion.one_minus_alphas_bar_sqrt)
    fig, axs = plt.subplots(2, num_divs+1, figsize=(28, 6))
    for i in range(num_divs + 1):
        cur_x = x_seq[i * int(num_steps/num_divs)].detach()
        axs[0, i if not reverse else num_divs-i].scatter(cur_x[:, 0], cur_x[:, 1],color='white',edgecolor='gray', s=5)
        axs[0, i if not reverse else num_divs-i].set_axis_off()
        axs[0, i if not reverse else num_divs-i].set_title('$q(\mathbf{x}_{'+str(int((num_divs-i)*(num_steps)/num_divs))+'})$')

        if heatmap:
            cur_df = pd.DataFrame(cur_x)
            sns.kdeplot(data=cur_df, x=0, y=1, fill=True, thresh=0, levels=100, ax=axs[1, i if not reverse else num_divs-i], cmap="mako")


def forward_diffusion(dataset, num_steps, plot=False, num_divs=10):
    """Applies forward diffusion to the dataset with the given number of steps

    Args:
        dataset (torch.Tensor): the dataset to be diffused
        num_steps (int): the step size for the addition of noise to the data
        plot (bool): true if you want to plot the data with the noise added
        num_divs (int): number of plots to make. Only applicable if plot=True
    """
    diffusion = Diffusion(num_steps)

    if plot:
        # Visualize the forward process
        visualize_forward(dataset, num_steps, num_divs, diffusion)

    return diffusion

def reverse_diffusion(dataset, diffusion, training_time_steps=0, plot=False, num_divs=10, show_heatmap=False, model=None):
    """Applies reverse diffusion to a dataset

    Args:
        dataset (torch.Tensor): the dataset to be used
        diffusion (class: Diffusion): a diffusion model class encapsulating proper constants for forward diffusion
                                Constants calculated from num_steps input to class constructor
        training_time_steps (int): number of training steps to remove noise.  Default is step_size from diffusion class
        plot (bool): true if you want to plot the data showing the removal of the noise
        num_divs (int): number of plots to show. Default is 10 and only applicable if plot=True
        model (ConditionalModel): optional argument to train a previously defined model

    Returns:
        model (ConditionalModel): the trained model
    """
    # Load variables from diffusion class
    num_steps = diffusion.num_steps
    betas = diffusion.betas
    alphas = diffusion.alphas
    alphas_prod = diffusion.alphas_prod
    alphas_bar_sqrt = diffusion.alphas_bar_sqrt
    one_minus_alphas_bar_sqrt = diffusion.one_minus_alphas_bar_sqrt

    if training_time_steps == 0:
        training_time_steps = num_steps

    # If no model given, create new one
    if model == None:
        model = ConditionalModel(num_steps, dataset.size()[1])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Create EMA model
    ema = EMA(0.9)
    ema.register(model)

    batch_size = 128
    # Define training loop until f1 score for separability between real and fake goes below threshold
    f1, t = 1.0, 0
    while f1 > .6 and t < 10000:            # New loop for threshold or max t
    # for t in range(training_time_steps):  # Previous loop for x amount of backwards steps
        permutation = torch.randperm(dataset.size()[0])
        for i in range(0, dataset.size()[0], batch_size):
            # Retrieve current batch
            indices = permutation[i:i+batch_size]
            batch_x = dataset[indices]
            # Compute the loss
            loss = noise_estimation_loss(model, batch_x,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,num_steps)
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
        if t % 1000 == 0:
            fake = get_model_output(model, dataset.shape[1], diffusion, num_to_gen=1000)
            _, _, _, f1 = separability(dataset, fake, train_test_ratio=.7, printStats=False)
        print(f'Training Steps: {t}\tLoss: {round(loss.item(), 4)}\tF1: {round(f1, 4)}\r', end='')
        t += 1
    if plot:
        plt.show()

    return model

def use_model(model, dataset, diffusion, t):
    """Takes in a trained diffusion model and creates n datapoints from Gaussian noise

    Args:
        model (ConditionalModel): a trained diffusion model
        dataset (torch.Tensor): the dataset shape to model from
        diffusion (Diffusion): the diffusion variables for the denoising process
        t (int): the number of time steps to remove from the noise

    Returns:
        data (torch.Tensor): the generated data
    """
    output = p_sample(model, dataset, t, diffusion.alphas, diffusion.betas, diffusion.one_minus_alphas_bar_sqrt)
    return output

def reverse_categorical_diffusion(discrete, features, diffusion, k, feature_indices, batch_size = 128, lr=1e-3, training_time_steps=0, plot=False, num_divs=10, show_heatmap=False, model=None):
    """Applies reverse diffusion to a dataset

    NOTE: In order to use, must modify loss function to use only categorical model
    Args:
        discrete (torch.Tensor): the dataset to be used
        features (list<strings>): a list of the feature names for the dataset
        diffusion (class: Diffusion): a diffusion model class encapsulating proper constants for forward diffusion
                                Constants calculated from num_steps input to class constructor
        k (int): number of total classes across all features
        feature_indices (list<tuples>): a list of the indices for all the features
        training_time_steps (int): number of training steps to remove noise.  Default is step_size from diffusion class
        plot (bool): true if you want to plot the data showing the removal of the noise
        num_divs (int): number of plots to show. Default is 10 and only applicable if plot=True
        model (ConditionalModel): optional argument to train a previously defined model
    Returns:
        model (ConditionalModel): the trained model
    """
    # Load variables from diffusion class
    num_steps = diffusion.num_steps
    betas = diffusion.betas
    alphas = diffusion.alphas
    alphas_prod = diffusion.alphas_prod
    alphas_bar_sqrt = diffusion.alphas_bar_sqrt
    one_minus_alphas_bar_sqrt = diffusion.one_minus_alphas_bar_sqrt

    if training_time_steps == 0:
        training_time_steps = num_steps

    # If no model given, create new one
    if model == None:
        model = ConditionalMultinomialModel(num_steps, discrete.shape[1])

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Create EMA model
    ema = EMA(0.9)
    ema.register(model)

    # Only tracked for graphing loss afterwards
    loss_list, prob_list = [], []

    for t in range(training_time_steps):
        permutation_discrete = torch.randperm(discrete.shape[0])
        for i in range(0, discrete.shape[0], batch_size):
            # Retrieve current batch
            indices_discrete = permutation_discrete[i:i+batch_size]
            batch_x_discrete = discrete[indices_discrete]
            # One hot encoding
            batch_x_discrete = to_one_hot(batch_x_discrete, k, feature_indices)
            # Compute the loss
            loss = multinomial_diffusion_noise_estimation(model, batch_x_discrete, diffusion, k, feature_indices)
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
        p = get_discrete_model_output(model, k, 1, feature_indices).squeeze(0)
        prob_list.append(p)
        print(f'Training Steps: {t}\tLoss: {round(loss.item(), 8)}\r', end='')
        loss_list.append(loss.item())

    return model, loss_list, prob_list

############################
###  CODE TO WORK ON FOR ###
###   TABULAR DIFFUSION  ###
############################
def reverse_tabular_diffusion(discrete, continuous, features, diffusion, k, feature_indices, batch_size = 128, lr=1e-3, training_time_steps=0, plot=False, num_divs=10, show_heatmap=False, model=None):
    """Applies reverse diffusion to a dataset

    Args:
        discrete (torch.Tensor): the discrete features
        continuous (torch.Tensor): the continuous features
        features (list<strings>): a list of the feature names for the dataset
        diffusion (class: Diffusion): a diffusion model class encapsulating proper constants for forward diffusion
                                Constants calculated from num_steps input to class constructor
        k (int): number of total classes across all features
        feature_indices (list<tuples>): a list of the indices for all the features
        training_time_steps (int): number of training steps to remove noise.  Default is step_size from diffusion class
        plot (bool): true if you want to plot the data showing the removal of the noise
        num_divs (int): number of plots to show. Default is 10 and only applicable if plot=True
        model (ConditionalModel): optional argument to train a previously defined model

    Returns:
        model (ConditionalModel): the trained model
    """
    # Load variables from diffusion class
    num_steps = diffusion.num_steps
    betas = diffusion.betas
    alphas = diffusion.alphas
    alphas_prod = diffusion.alphas_prod
    alphas_bar_sqrt = diffusion.alphas_bar_sqrt
    one_minus_alphas_bar_sqrt = diffusion.one_minus_alphas_bar_sqrt

    if training_time_steps == 0:
        training_time_steps = num_steps

    # If no model given, create new one
    if model == None:
        model = ConditionalMultinomialModel(num_steps, discrete.shape[1])

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Create EMA model
    ema = EMA(0.9)
    ema.register(model)

    # Only tracked for graphing loss afterwards
    loss_list, prob_list = [], []

    for t in range(training_time_steps):
        permutation_discrete = torch.randperm(discrete.shape[0])
        permutation_continuous = torch.randperm(continuous.shape[0])
        for i in range(0, discrete.shape[0], batch_size):
            # Retrieve current batch
            indices_discrete = permutation_discrete[i:i+batch_size]
            indices_continuous = permutation_continuous[i:i+batch_size]
            batch_x_discrete = discrete[indices_discrete]
            batch_x_continuous = continuous[indices_continuous]
            # One hot encoding
            batch_x_discrete = to_one_hot(batch_x_discrete, k, feature_indices)
            # Compute the loss
            multinomial_loss = multinomial_diffusion_noise_estimation(model, batch_x_discrete, batch_x_continuous, diffusion, k, feature_indices)
            continuous_loss = noise_estimation_loss(model, batch_x_continuous, batch_x_discrete, feature_indices, k, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            loss = multinomial_loss + continuous_loss
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
        _, p = get_discrete_model_output(model, k, 1000, feature_indices, continuous)
        prob_list.append(p.squeeze(0))
        print(f'Training Steps: {t}\tLoss: {round(loss.item(), 8)}\r', end='')
        loss_list.append(loss.item())

    return model, loss_list, prob_list

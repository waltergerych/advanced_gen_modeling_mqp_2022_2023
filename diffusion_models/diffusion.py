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
    """Function to add t time steps of noise to data x
    
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

def q_x_cat(x_0, t, model, K):
    """Function to add t time steps of noise to categorical data x
    
    Args:
        x_0 (torch.Tensor): Categorical data to add noise to
        t (torch.Tensor): the number of time steps to add
        model (class: Diffusion): a diffusion model class encapsulating proper constants for forward diffusion
                                Constants calculated from num_steps input to class constructor
        K (torch.Tensor): Tensor of size 1 representing the number of classes the categorical variable x can take

    Returns:
        (torch.Tensor): the data with the noise added to it
    """
    alphas_t = extract(model.alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(model.one_minus_alphas_bar_sqrt, t, x_0)
    cat = (alphas_t * x_0 + alphas_1_m_t) / K
    print(cat)
    return cat

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

def reverse_diffusion(dataset, diffusion, training_time_steps=0, plot=False, num_divs=10, show_heatmap=False):
    """Applies reverse diffusion to a dataset

    Args:
        dataset (torch.Tensor): the dataset to be used
        diffusion (class: Diffusion): a diffusion model class encapsulating proper constants for forward diffusion
                                Constants calculated from num_steps input to class constructor
        training_time_steps (int): number of training steps to remove noise.  Default is step_size from diffusion class
        plot (bool): true if you want to plot the data showing the removal of the noise
        num_divs (int): number of plots to show. Default is 10 and only applicable if plot=True

    Returns:
        model (torch.Tensor.state.dict): the trained model
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

    model = ConditionalModel(num_steps, dataset.size()[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Create EMA model
    ema = EMA(0.9)
    ema.register(model)

    batch_size = 128
    for t in range(training_time_steps):
        # X is a torch Variable
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
        if (t % int(training_time_steps/(num_divs*2)) == 0):
            # Print every 5% update
            print(f'{int((t/training_time_steps)*100) if t != 0 else 0}% done \t loss: {loss}')
            # Draw plots every num_divs steps (default 10)
            if plot and t % (training_time_steps/num_divs) == 0:
                visualize_backward(model, dataset, num_steps, num_divs, diffusion, heatmap=show_heatmap)
                # visualize_heatmap(model, dataset, num_steps, num_divs, diffusion)
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
    
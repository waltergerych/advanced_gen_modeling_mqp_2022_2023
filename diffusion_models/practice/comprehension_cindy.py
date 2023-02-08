"""
This file will have snippets of code in which you will need to 
fill in missing code and/or explain what some code does.  Some
code for functions and classes is in 'functions.py' which you 
can use to help
"""

from functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#######################
# 1. Forward diffusion
# The following is code for forward diffusion:

def forward_diffusion(num_steps: int):
    diffusion = Diffusion(num_steps)
    return diffusion

# What does this code do and why is it so short? 

# This snippet of code declares the diffusion model with the given number of timesteps. 
# It is so short because the Diffusion class definition sets up all the necessary parameters, including 
# the variance schedule, which is the only user choice required for the forward process.  

#######################
# 2. Reverse diffusion
# Complete the missing code for declaring the model for reverse diffusion

def reverse_diffusion(dataset: torch.Tensor, diffusion: Diffusion):

    model = ConditionalModel(diffusion.num_steps, dataset.shape[0])    # Replace None with correct arguments
                            # input, output sizes? how to get from dataset? what does dataset look like?

    # For reverse process, you need to choose model architecture.
    # Input and output must have same dimensionality.
    # What is some_int? Is it num_embeddings? (in the declaration for CondiionalModel in functions.py line 42)
    # num_steps
     

#######################
# 3. Graphing/visualization
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
    x_seq = p_sample_loop(model, dataset.shape, num_steps, diffusion.alphas, diffusion.betas, diffusion.one_minus_alphas_bar_sqrt)     # TODO
    fig, axs = plt.subplots(2, num_divs+1, figsize=(28, 6))
    for i in range(num_divs + 1):
        cur_x = ___.detach()            # TODO
        axs[0, i if not reverse else num_divs-i].scatter(cur_x[:, 0], cur_x[:, 1],color='white',edgecolor='gray', s=5)
        axs[0, i if not reverse else num_divs-i].set_axis_off()
        axs[0, i if not reverse else num_divs-i].set_title('$q(\mathbf{x}_{'+str(int((num_divs-i)*(num_steps)/num_divs))+'})$')

        if heatmap:
            cur_df = pd.DataFrame(cur_x)
            sns.kdeplot(data=cur_df, x=0, y=1, fill=True, thresh=0, levels=100, ax=axs[1, i if not reverse else num_divs-i], cmap="mako")

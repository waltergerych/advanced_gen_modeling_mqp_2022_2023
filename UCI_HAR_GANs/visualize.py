# External libraries
import torch
import matplotlib.pyplot as plt
import numpy as np

def make_histograms(data):
    """ Make a histogram for every feature in the provided data set and save in a folder

    Args:
        data (torch.Tensor): the data to visualize with histograms

    Returns:
        None
    """
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
    # for col in range(len(data[0])):
    for col in range(20):
        hist = torch.histc(data[0:, col], min=-1, max=1, bins=10, out=None)

        x = np.linspace(-1, 1, 10)
        plt.cla()
        plt.bar(x, hist, align='center', color=colors[col%6])
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.savefig(f'./histograms/fake/{col}.png')

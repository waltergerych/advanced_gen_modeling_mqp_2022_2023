# External libraries
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import rel_entr
from scipy.spatial import distance
from sklearn.decomposition import PCA

# was referencing https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# need change param
def perform_pca(data):
    """ Perform a principal component analysis (PCA) for each 

    Args:
        data (torch.Tensor): the data to run PCA testing on

    Returns:
        None
    """

    # fig = plt.figure(figsize = (8,8))
    # ax = fig.add_subplot(1,1,1) 
    # ax.set_xlabel('Principal Component 1', fontsize = 15)
    # ax.set_ylabel('Principal Component 2', fontsize = 15)
    # ax.set_title('2 component PCA', fontsize = 20)
    # targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    # colors = ['r', 'g', 'b']
    # for target, color in zip(targets,colors):
    #     indicesToKeep = finalDf['target'] == target
    #     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
    #             , finalDf.loc[indicesToKeep, 'principal component 2']
    #             , c = color
    #             , s = 50)
    # ax.legend(targets)
    # ax.grid()

    # standardize data
    pca = PCA(n_components=6)
    pca.fit(data)

    # PCA projection to 2D

    # visualize the 2D

    #
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)


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
        # plt.savefig(f'./histograms/fake/{col}.png')


def divergence(real, fake):
    """ Calculates the Kullback-Leibler (KL) Divergence score and the Jenson-Shannon (JS) 
        Divergence score between real and fake data distributions and displays a graph of 
        the scores for each feature's distributions

    Args:
        real (torch.Tensor): the tensor of real data
        fake (torch.Tensor): the tensor of fake data

    Returns:
        None
    """
    kl_scores, js_scores = [], []
    for col in range(len(real[0])):
    # for col in range(20):
        real_dist = torch.histc(real[0:, col], min=-1, max=1, bins=10, out=None)
        fake_dist = torch.histc(fake[0:, col], min=-1, max=1, bins=10, out=None)

        real_dist = torch.div(real_dist, torch.sum(real_dist))
        fake_dist = torch.div(fake_dist, torch.sum(fake_dist))

        kl_divergence = sum(rel_entr(fake_dist.tolist(), real_dist.tolist()))
        kl_scores.append(kl_divergence)
        js_divergence = distance.jensenshannon(fake_dist.tolist(), real_dist.tolist())
        js_scores.append(js_divergence)

    print("KL Divergence:\n" + str(kl_scores))
    print("\nJS Divergence:\n" + str(js_scores))
    x = range(len(real[0]))

    plt.plot(x, kl_scores)
    plt.xlabel('Feature Space')
    plt.ylabel('KL Divergence')
    plt.show()

    plt.plot(x, js_scores)
    plt.xlabel('Feature Space')
    plt.ylabel('JS Divergence')
    plt.show()


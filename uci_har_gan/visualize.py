# External libraries
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.spatial import distance
from scipy import stats
from sklearn.decomposition import PCA

# was referencing https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# need change param
def perform_pca(real, fake):
    """ Perform a principal component analysis (PCA) on the data and visualize on a 2D plane

    Args:
        real (torch.Tensor): the real data for pca
        fake (torch.Tensor): the generated data for pca

    Returns:
        None
    """
    labels = np.concatenate((np.ones(len(real)), np.zeros(len(fake))))

    data = torch.cat((real, fake), 0)

    pca = PCA(n_components=2)
    components = pca.fit_transform(data.detach().numpy())

    # PCA projection to 2D
    df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])

    # visualize the 2D
    plt.figure(figsize = (8,8))
    plt.scatter(df['PC1'], df['PC2'], c=labels, label=labels, alpha=.5)
    plt.legend(['Real', 'Fake'])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA with Real and Fake Data")
    plt.show()

    print("Explained variance ratio: " + str(pca.explained_variance_ratio_))
    print("Singular values: " + str(pca.singular_values_))


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


def divergence(real, real_label, fake, fake_label):
    """ Calculates the Kullback-Leibler (KL) Divergence score and the Jenson-Shannon (JS) 
        Divergence score between real and fake data distributions with and without the classes,
        calculates the lower and upper bound for divergence. The function then prints the 
        resulting comparisons and graphs the bounds for each class to visualize easier

    Args:
        real (torch.Tensor): the tensor of real data
        real_label (torch.Tensor): the class labels for the real data
        fake (torch.Tensor): the tensor of fake data
        fake_labels (torch.Tensor): the class labels for the fake data

    Returns:
        None
    """
    classes = ['WALKING', 'U-STAIRS', 'D-STAIRS', 'SITTING', 'STANDING', 'LAYING']
    # Divergence without classes
    real_dist = torch.histc(real, min=-1, max=1, out=None)
    fake_dist = torch.histc(fake, min=-1, max=1, out=None)

    real_dist = torch.div(real_dist, torch.sum(real_dist)).tolist()
    fake_dist = torch.div(fake_dist, torch.sum(fake_dist)).tolist()

    kl_fake_divergence = sum(rel_entr(fake_dist, real_dist))
    js_fake_divergence = distance.jensenshannon(fake_dist, real_dist)

    # Calculate upper bound for divergence score with KL(gaussian || real)
    x = np.linspace(-1, 1, 100)
    gauss = stats.norm.pdf(x, 0, 1)
    kl_upper_bound = sum(rel_entr(gauss, real_dist))
    js_upper_bound = distance.jensenshannon(gauss, real_dist)
    
    # Calculate lower bound for divergence score with KL(real_sample || real) where 
    # real_sample takes a random sample of half the real data
    rand_idx = np.random.permutation(real.size(0))
    real_sample = real[rand_idx[:int(real.size(0)/2)],:]

    sample_dist = torch.histc(real_sample, min=-1, max=1, out=None)
    sample_dist = torch.div(sample_dist, torch.sum(sample_dist)).tolist()
    kl_lower_bound = sum(rel_entr(sample_dist, real_dist))
    js_lower_bound = distance.jensenshannon(sample_dist, real_dist)

    print("Divergence without Classes")
    print("\t\t\tLower Bound\t\tDivergence\t\tUpper Bound")
    print("\tKL\t" + str(kl_lower_bound) + "\t" + str(kl_fake_divergence) + "\t" + str(kl_upper_bound))
    print("\tJS\t" + str(js_lower_bound) + "\t" + str(js_fake_divergence) + "\t" + str(js_upper_bound))
    

    # Divergence between class distributions
    kl_scores, js_scores = [], []
    kl_lower_bounds, kl_upper_bounds, js_lower_bounds, js_upper_bounds= [], [], [], []
    for c in range(len(classes)):
        real_temp = real[(real_label == c).nonzero().squeeze(1)]
        fake_temp = fake[(fake_label == c).nonzero().squeeze(1)]

        real_dist = torch.histc(real_temp, min=-1, max=1, out=None)
        fake_dist = torch.histc(fake_temp, min=-1, max=1, out=None)

        real_dist = torch.div(real_dist, torch.sum(real_dist)).tolist()
        fake_dist = torch.div(fake_dist, torch.sum(fake_dist)).tolist()

        kl_divergence = sum(rel_entr(fake_dist, real_dist))
        kl_scores.append(kl_divergence)
        js_divergence = distance.jensenshannon(fake_dist, real_dist)
        js_scores.append(js_divergence)

        # Calculate upper bound for divergence score with KL(gaussian || real)
        kl_upper_bounds.append(sum(rel_entr(gauss, real_dist)))
        js_upper_bounds.append(distance.jensenshannon(gauss, real_dist))
        
        # Calculate lower bound for divergence score with KL(real_sample || real) where 
        # real_sample takes a random sample of half the real data for the class
        rand_idx = np.random.permutation(real_temp.size(0))
        real_sample = real_temp[rand_idx[:int(real_temp.size(0)/2)],:]

        sample_dist = torch.histc(real_sample, min=-1, max=1, out=None)
        sample_dist = torch.div(sample_dist, torch.sum(sample_dist)).tolist()
        kl_lower_bounds.append(sum(rel_entr(sample_dist, real_dist)))
        js_lower_bounds.append(distance.jensenshannon(sample_dist, real_dist))

    print("\nDivergence with Classes")
    print("\t\t\tLower Bound\t\tDivergence\t\tUpper Bound")
    for i, c in enumerate(classes):
        print(c)
        print("\tKL\t" + str(kl_lower_bounds[i]) + "\t" + str(kl_scores[i]) + "\t" + str(kl_upper_bounds[i]))
        print("\tJS\t" + str(js_lower_bounds[i]) + "\t" + str(js_scores[i]) + "\t" + str(js_upper_bounds[i]))

    plt.scatter(classes, np.log2(kl_scores), c='green')
    plt.scatter(classes, np.log2(kl_lower_bounds), c='black')
    plt.scatter(classes, np.log2(kl_upper_bounds), c='black')
    plt.xlabel('Classes')
    plt.ylabel('Log of KL Divergence')
    plt.title('KL Divergences')
    plt.show()

    plt.scatter(classes, js_scores, c='green')
    plt.scatter(classes, js_lower_bounds, c='black')
    plt.scatter(classes, js_upper_bounds, c='black')
    plt.xlabel('Classes')
    plt.ylabel('JS Divergence')
    plt.title('JS Divergences')
    plt.show()


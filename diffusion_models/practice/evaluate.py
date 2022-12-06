import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import joblib

from sklearn.decomposition import PCA
from utils import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def perform_pca(real, fake, title=None):
    """ Perform a principal component analysis (PCA) on the data and visualize on a 2D plane

    Args:
        real (torch.Tensor): the real data for pca
        fake (torch.Tensor): the generated data for pca

    Returns:
        None
    """
    if title == None:
        title = 'PCA With Real and Fake Data'

    labels = np.concatenate((np.ones(len(real)), np.zeros(len(fake))))
    data = torch.cat((real, fake), 0)

    pca = PCA(n_components=2)
    components = pca.fit_transform(data.detach().numpy())

    # PCA projection to 2D
    df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])

    # visualize the 2D
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    scatter = plt.scatter(df['PC1'], df['PC2'], c=labels, alpha=.8, marker='.')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Fake', 'Real'])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    # plt.savefig(f'./results/{title}.png')
    # plt.show()

def perform_pca_har_classes(data, labels, classes):
    """ Perform a principal component analysis (PCA) on the data and visualize on a 2D plane

    Args:
        data (torch.Tensor): the data with all classes for pca
        labels (torch.Tensor): the class labels for the data
        classes (list<strings>): the names of the classes labels

    Returns:
        None
    """
    pca = PCA(n_components=2)
    components = pca.fit_transform(data.detach().numpy())

    # PCA projection to 2D
    df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])

    # visualize the 2D
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('white')
    scatter = plt.scatter(df['PC1'], df['PC2'], c=labels, alpha=.8, marker='.')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA")
    plt.show()

def pca_class_real_and_fake(real_data, real_labels, fake_data, fake_labels, classes):
    """ Perform a principal component analysis (PCA) on real and fake data and shows class subcategories

    Args:
        data (torch.Tensor): the data with all classes for pca
        labels (torch.Tensor): the class labels for the data
        classes (list<strings>): the names of the classes labels

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('white')

    # Project real to 2D
    pca = PCA(n_components=2)
    components = pca.fit_transform(real_data.detach().numpy())
    df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])

    # Visualize real 2D
    scatter = plt.scatter(df['PC1'], df['PC2'], c=real_labels, alpha=.5, marker='|')

    # Project fake to 2D
    pca = PCA(n_components=2)
    components = pca.fit_transform(fake_data.detach().numpy())
    df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])

    # Visualize fake 2D
    scatter = plt.scatter(df['PC1'], df['PC2'], c=fake_labels, alpha=.3, marker='_')

    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA with Real and Fake Data")
    plt.show()

def graph_two_features(real, fake, noise=None):
    """ Graphs real and fake data with only two features

    Args:
        real (torch.Tensor): the real data with two features
        fake (torch.Tensor): the generated data with two features
        [optional] noise (torch.Tensor): if supplied, can graph the real data with noise added to it

    Returns:
        None
    """
    if noise != None:
        labels = np.concatenate((np.ones(len(real)), np.zeros(len(fake)), np.ones(len(noise))*2))
        data = torch.cat((real, fake, noise), 0).detach().numpy()
        label_names = ['Fake', 'Real', 'Real + Noise']
    else:
        labels = np.concatenate((np.ones(len(real)), np.zeros(len(fake))))
        data = torch.cat((real, fake), 0).detach().numpy()
        label_names = ['Fake', 'Real']

    # PCA projection to 2D
    df = pd.DataFrame(data=data, columns=['X1', 'X2'])

    # visualize the 2D
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    scatter = plt.scatter(df['X1'], df['X2'], c=labels, alpha=.8, marker='.')
    plt.legend(handles=scatter.legend_elements()[0], labels=label_names)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Real and Fake Data")
    # plt.show()

def make_histograms(data, num_features):
    """ Make a histogram for every feature in the provided data set and save in a folder

    Args:
        data (torch.Tensor): the data to visualize with histograms

    Returns:
        None
    """
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
    for col in range(num_features):
        hist = torch.histc(data[0:, col], min=-1, max=1, bins=10, out=None)

        x = np.linspace(-1, 1, 10)
        plt.cla()
        plt.bar(x, hist, align='center', color=colors[col%6])
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.show()
        # plt.savefig(f'./histograms/fake/{col}.png')

def build_binary_classifier(data, labels, classes, class_index):
    """Builds a random forest classifier to decide whether or not data belongs to the specified class

    Ex: Input:data --> Model --> Output: Walking? Yes

    Args:
        data (torch.Tensor): the data to be used for training the model
        labels (torch.Tensor): the class labels for the data
        classes (list): a list of all of the class labels
        class_index (int): the index for the desired class from classes in which the binary classifier will discriminate on

    Returns:
        model (RandomForestClassifier): the trained classifier model
    """
    model = RandomForestClassifier(max_depth=10)
    train_labels = torch.eq(labels, torch.ones(data.size(0))*class_index).int()
    model.fit(data.detach().numpy(), train_labels)
    # joblib.dump(model, f'./classifiers/rf_{class_index}.joblib', compress=3)

    return model

def load_binary_classifier(path):
    """Loads a classifier from the specified path

    Args:
        path (String): the relative path where the model is saved
    """
    model = joblib.load(path)
    return model

def test_binary_classifier(classifier, data, labels, classes, class_index):
    """Runs machine evaluation using a binary classifier to decide whether data belongs to the specified class

    Args:
        classifier (*some sklearn classifier*): a trained classifier to be used
        data (torch.Tensor): the data to be tested
        labels (torch.Tensor): the labels for the data
        classes (list): a list of all possible classes
        class_index (int): the index of the desired class to test for
    """
    test_labels = torch.eq(labels, torch.ones(data.size(0))*class_index).int()
    data = data.detach().numpy()
    pred = classifier.predict(data)

    print(confusion_matrix(test_labels, pred))
    print("Accuracy\t" + str(classifier.score(data, test_labels)))



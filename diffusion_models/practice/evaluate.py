import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import joblib
import seaborn as sns

from sklearn.decomposition import PCA
from utils import *
import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


TITLE_FONT_SIZE = 15
HEATMAP_ALPHA = 0.4
CLASSIFIER_DATA_PATH = "classifier_data"


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
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])

    # Get df for just real data
    # real_pca = PCA(n_components=2)
    # real_components = real_pca.fit_transform(real.detach().numpy())
    # real_data_df = pd.DataFrame(data=real_components, columns=['PC1', 'PC2'])
    real_df = pca_df.head(len(real))

    # Get df for just fake data
    # fake_pca = PCA(n_components=2)
    # fake_components = fake_pca.fit_transform(fake.detach().numpy())
    # fake_data_df = pd.DataFrame(data=fake_components, columns=['PC1', 'PC2'])
    fake_df = pca_df.tail(len(fake))




    fig = plt.figure(figsize=(12, 10))

    # Make top margin smaller on figure and add some padding between graphs
    fig.subplots_adjust(top = 0.9, hspace=0.3)

    # PCA Graph (Upper left)
    ax = fig.add_subplot(2, 2, 1)
    ax.set_facecolor('white')
    scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=labels, alpha=.8, marker='.')
    ax.legend(handles=scatter.legend_elements()[0], labels=['Fake', 'Real'])
    ax.set_xlabel("PC1", fontsize=TITLE_FONT_SIZE)
    ax.set_ylabel("PC2", fontsize=TITLE_FONT_SIZE)
    ax.set_title(f'PCA for class {title}', fontsize=TITLE_FONT_SIZE)
    # Get axis ranges 
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Heatmap of PCA (Upper right)
    ax = fig.add_subplot(2, 2, 2)
    sns.kdeplot(data=pca_df, x='PC1', y='PC2', fill=True, thresh=0, levels=100, ax=ax, cmap="mako")
    ax.set_xlabel("PC1", fontsize=TITLE_FONT_SIZE)
    ax.set_ylabel("PC2", fontsize=TITLE_FONT_SIZE)
    ax.set_title(f'Heatmap for class {title}', fontsize=TITLE_FONT_SIZE)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Heatmap of just real data (Lower left)
    ax = fig.add_subplot(2, 2, 3)
    sns.kdeplot(data=real_df, x='PC1', y='PC2', fill=True, thresh=0, levels=100, ax=ax, cmap="mako")
    ax.set_xlabel("PC1", fontsize=TITLE_FONT_SIZE)
    ax.set_ylabel("PC2", fontsize=TITLE_FONT_SIZE)
    ax.set_title(f'Heatmap for real {title} data', fontsize=TITLE_FONT_SIZE)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Heatmap of just fake data (Lower right)
    ax = fig.add_subplot(2, 2, 4)
    sns.kdeplot(data=fake_df, x='PC1', y='PC2', fill=True, thresh=0, levels=100, ax=ax, cmap="mako")
    ax.set_xlabel("PC1", fontsize=TITLE_FONT_SIZE)
    ax.set_ylabel("PC2", fontsize=TITLE_FONT_SIZE)
    ax.set_title(f'Heatmap for fake {title} data', fontsize=TITLE_FONT_SIZE)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    


def pca_with_classes(real_data, real_labels, fake_data, fake_labels, classes, overlay_heatmap=True):
    """ Perform a principal component analysis (PCA) on real and fake data and shows class subcategories

    Args:
        data (torch.Tensor): the data with all classes for pca
        labels (torch.Tensor): the class labels for the data
        classes (list<strings>): the names of the classes labels

    Returns:
        None
    """
    # Combine data
    data = torch.cat([real_data, fake_data], dim=0)

    # Fit PCA on combined data
    pca = PCA(n_components=2)
    components = pca.fit_transform(data.detach().numpy())

    # Separate into real and fake classes again
    real_components = components[:real_data.shape[0]]
    fake_components = components[real_data.shape[0]:]

    # PCA projection to 2D
    real = pd.DataFrame(data=real_components, columns=['PC1', 'PC2'])
    fake = pd.DataFrame(data=fake_components, columns=['PC1', 'PC2'])


    # First figure, PCA plots
    fig = plt.figure(figsize=(16, 8))

    # Real data PCA all classes (Upper left)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_facecolor('white')
    scatter = plt.scatter(real['PC1'], real['PC2'], c=real_labels, alpha=.8, marker='.')
    ax.legend(handles=scatter.legend_elements()[0], labels=classes)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA with Real Data")
    # Get axis ranges 
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Overlay heatmap
    if overlay_heatmap: sns.kdeplot(data=real, x='PC1', y='PC2', fill=True, thresh=0, levels=100, ax=ax, cmap="mako", alpha=HEATMAP_ALPHA)

    # Fake data PCA all classes (Upper right)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_facecolor('white')
    scatter = plt.scatter(fake['PC1'], fake['PC2'], c=fake_labels, alpha=.8, marker='.')
    ax.legend(handles=scatter.legend_elements()[0], labels=classes)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA with Fake Data")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Overlay heatmap
    if overlay_heatmap: sns.kdeplot(data=fake, x='PC1', y='PC2', fill=True, thresh=0, levels=100, ax=ax, cmap="mako", alpha=HEATMAP_ALPHA)

    plt.show()

    if not overlay_heatmap:
        # Second figure, heatmaps of PCA plots
        fig = plt.figure(figsize=(16, 8))

        # Heatmap for real PCA all classes (Lower left)
        ax = fig.add_subplot(1, 2, 1)
        sns.kdeplot(data=real, x='PC1', y='PC2', fill=True, thresh=0, levels=100, ax=ax, cmap="mako")
        ax.set_xlabel("PC1", fontsize=TITLE_FONT_SIZE)
        ax.set_ylabel("PC2", fontsize=TITLE_FONT_SIZE)
        ax.set_title(f'Heatmap for real data', fontsize=TITLE_FONT_SIZE)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Heatmap for fake PCA all classes (Lower left)
        ax = fig.add_subplot(1, 2, 2)
        sns.kdeplot(data=fake, x='PC1', y='PC2', fill=True, thresh=0, levels=100, ax=ax, cmap="mako")
        ax.set_xlabel("PC1", fontsize=TITLE_FONT_SIZE)
        ax.set_ylabel("PC2", fontsize=TITLE_FONT_SIZE)
        ax.set_title(f'Heatmap for fake data', fontsize=TITLE_FONT_SIZE)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

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

def score(labels, pred):
    """Calculates accuracy, recall, precision, and f1 based on true labels and predicted labels

    Args:
        labels (list): the list of true labels
        pred (list): the list of predicted labels

    Returns:
        accuracy (float)
        precision (float)
        recall (float)
        f1-score (float)
    """
    cm = confusion_matrix(labels, pred)
    tn, fp, fn, tp = cm.ravel()
    tn, fp, fn, tp
    accuracy = accuracy_score(labels, pred)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = (2 * precision * recall) / (recall + precision)

    return accuracy, precision, recall, f1

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

    Returns:
        accuracy (float)
        precision (float)
        recall (float)
        f1-score (float)
    """
    test_labels = torch.eq(labels, torch.ones(data.size(0))*class_index).int()
    data = data.detach().numpy()
    pred = classifier.predict(data)

    accuracy, precision, recall, f1 = score(test_labels, pred)

    print(confusion_matrix(test_labels, pred))
    print(f'Accuracy:\t{accuracy}')
    print(f'F1:\t\t{f1}')

    return accuracy, precision, recall, f1

def build_multiclass_classifier(data, labels):
    """Builds a mulitclass classifier to predict whether data is one of:
            WALKING
            DOWNSTAIRS
            UPSTAIRS
            SITTING
            STANDING
            LAYING
    Args:
        data (torch.Tensor): the data for the classifier to be trained on
        labels (torch.Tensor): the labels for the data
    """
    model = RandomForestClassifier(max_depth=10)

    model.fit(data.detach().numpy(), labels)

    return model

def test_multiclass_classifier(model, data, labels):
    """Tests a multiclass classifier to predict the class data

    Args:
        model (*some sklearn classifier*): a trained sklearn classifier
        data (torch.Tensor): the test data to predict the class label for
        labels (torch.Tensor): the true labels for the data

    Returns:
        accuracy (float)
        precision (float)
        recall (float)
        f1-score (float)
    """
    data = data.detach().numpy()
    pred = model.predict(data)
    print(classification_report(labels, pred))

def separability(real, fake, train_test_ratio):
    """Determines how separable real and fake data are from each other with a binary classifier

    Will print the output to the terminal

    Args:
        real (torch.Tensor): the real data
        fake (torch.Tensor): the fake data

    Returns:
        None
    """
    labels = torch.cat([torch.zeros(real.shape[0]), torch.ones(fake.shape[0])])
    data = torch.cat([real, fake])
    train_x, test_x, train_y, test_y = train_test_split(data.detach().numpy(), labels.detach().numpy(), test_size=train_test_ratio)

    model = RandomForestClassifier(max_depth=10)
    model.fit(train_x, train_y)

    pred = model.predict(test_x)
    accuracy, precision, recall, f1 = score(test_y, pred)
    print(f'Accuracy:\t{accuracy}')
    print(f'Precision:\t{precision}')
    print(f'Recall:\t{recall}')
    print(f'F1:\t\t{f1}')

    return model

def binary_machine_evaluation(dataset, labels, fake, fake_labels, classes, test_train_ratio, num_steps):
    """Evaluates data on binary classifiers

    Args:
        dataset (torch.Tensor): the real data
        labels (torch.Tensor): the labels for the real data
        fake (torch.Tensor): the fake data
        fake_labels (torch.Tensor): the labels for the fake data
        classes (list<strings>): a list of the possible classes
        test_train_ratio (float): a decimal value between 0 and 1 for the ratio of which to split test and train data
    """
    # Split into testing and training for classifier
    real_train_x, real_test_x, real_train_y, real_test_y = train_test_split(dataset, labels, test_size=test_train_ratio)
    fake_train_x, fake_test_x, fake_train_y, fake_test_y = train_test_split(fake, fake_labels, test_size=test_train_ratio)

    # Store data for writing to csv
    csv_data = {
        # Trained on real
        "real": {
            # Tested on real
            "real": {
                "acc": [],
                "precision": [],
                "recall": [],
                "f1": []
            },
            "fake": {
                "acc": [],
                "precision": [],
                "recall": [],
                "f1": []
            }
        },
        "fake": {
            "real": {
                "acc": [],
                "precision": [],
                "recall": [],
                "f1": []
            },
            "fake": {
                "acc": [],
                "precision": [],
                "recall": [],
                "f1": []
            }
        },
    }

    for i in range(len(classes)):
        print(f'\nEvaluating class {classes[i]}')

        # Train classifier on real data
        print('Testing classifier trained on real data')
        classifier = build_binary_classifier(real_train_x, real_train_y, classes, i)
        
        # Train on real, test on real
        print('Evaluating on real data')
        acc, precision, recall, f1 = test_binary_classifier(classifier, real_test_x, real_test_y, classes, i)
        csv_data["real"]["real"]["acc"].append(acc)
        csv_data["real"]["real"]["precision"].append(precision)
        csv_data["real"]["real"]["recall"].append(recall)
        csv_data["real"]["real"]["f1"].append(f1)

        # Train on real, test on fake
        print('Evaluating on fake data')
        acc, precision, recall, f1 = test_binary_classifier(classifier, fake_test_x, fake_test_y, classes, i)
        csv_data["real"]["fake"]["acc"].append(acc)
        csv_data["real"]["fake"]["precision"].append(precision)
        csv_data["real"]["fake"]["recall"].append(recall)
        csv_data["real"]["fake"]["f1"].append(f1)

        # Train classifier on diffusion model generated data
        print('Testing classifier trained on fake data')
        classifier = build_binary_classifier(fake_train_x, fake_train_y, classes, i)

        # Train on fake, test on real
        print('Evaluating on real data')
        acc, precision, recall, f1 = test_binary_classifier(classifier, real_test_x, real_test_y, classes, i)
        csv_data["fake"]["real"]["acc"].append(acc)
        csv_data["fake"]["real"]["precision"].append(precision)
        csv_data["fake"]["real"]["recall"].append(recall)
        csv_data["fake"]["real"]["f1"].append(f1)

        # Train on fake, test on fake
        print('Evaluating on fake data')
        acc, precision, recall, f1 = test_binary_classifier(classifier, fake_test_x, fake_test_y, classes, i)
        csv_data["fake"]["fake"]["acc"].append(acc)
        csv_data["fake"]["fake"]["precision"].append(precision)
        csv_data["fake"]["fake"]["recall"].append(recall)
        csv_data["fake"]["fake"]["f1"].append(f1)

    # Write data to csv
    for metric in csv_data["real"]["real"]:
        flag = "a"
        file_name = f"{CLASSIFIER_DATA_PATH}/binary_classifier_{metric}.csv"
        if not os.path.exists(file_name): flag = "w"
            
        with open(file_name, flag) as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write headers
            if flag == "w": csv_writer.writerow(["Trained On", "Tested Against", "Num Stpes", classes[0], classes[1], classes[2], classes[3], classes[4], classes[5]])
            for train in csv_data:
                for test in csv_data[train]:
                    csv_row = [train, test, num_steps]
                    for i in range(len(classes)):
                        csv_row.append(csv_data[train][test][metric][i])
                    csv_writer.writerow(csv_row)
                            
            

def multiclass_machine_evaluation(dataset, labels, fake, fake_labels, test_train_ratio):
    """Evaluates data multiclass classifiers and prints results

    Args:
        dataset (torch.Tensor): the real data
        labels (torch.Tensor): the labels for the real data
        fake (torch.Tensor): the fake data
        fake_labels (torch.Tensor): the labels for the fake data
        test_train_ratio (float): a decimal value between 0 and 1 for the ratio of which to split test and train data
    """
    # Split into testing and training for classifier
    real_train_x, real_test_x, real_train_y, real_test_y = train_test_split(dataset, labels, test_size=test_train_ratio)
    fake_train_x, fake_test_x, fake_train_y, fake_test_y = train_test_split(fake, fake_labels, test_size=test_train_ratio)

    # Train classifier on real data
    print('Testing classifier trained on real data')
    classifier = build_multiclass_classifier(real_train_x, real_train_y)
    
    print('Evaluating on real data')
    test_multiclass_classifier(classifier, real_test_x, real_test_y)

    print('Evaluating on fake data')
    test_multiclass_classifier(classifier, fake_test_x, fake_test_y)

    # Train classifier on diffusion model generated data
    print('Testing classifier trained on fake data')
    classifier = build_multiclass_classifier(fake_train_x, fake_train_y)

    print('Evaluating on real data')
    test_multiclass_classifier(classifier, real_test_x, real_test_y)

    print('Evaluating on fake data')
    test_multiclass_classifier(classifier, fake_test_x, fake_test_y)
    

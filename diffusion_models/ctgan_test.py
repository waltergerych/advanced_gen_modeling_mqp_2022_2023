
# Native libraries
import os

# Internal libraries
import evaluate as eval
import utils
from helper_plot import hdr_plot_style

# External libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split


def main():
    # set plot style
    hdr_plot_style()

    # load the datasets
    train_x, train_y = utils.load_data('../dataset/UCI_HAR_Dataset', 'train')
    test_x, test_y = utils.load_data('../dataset/UCI_HAR_Dataset', 'test')

    classes = ['WALKING', 'U-STAIRS', 'D-STAIRS', 'SITTING', 'STANDING', 'LAYING']

    # define the number of features from the dataset to use. Must be 561 or less
    NUM_FEATURES = 15
    NUM_STEPS = 10000
    EPOCHS = 40000
    NUM_SAMPLE = 300
    TE_TR_RATIO = .3

    # use feature selection to select most important features
    feature_selector = SelectKBest(k=NUM_FEATURES)
    selection = feature_selector.fit(train_x, train_y)

    # select k best features on train data
    train_data = torch.tensor(selection.transform(train_x))
    discrete_tr = torch.from_numpy(np.loadtxt(os.path.join('../dataset/UCI_HAR_Dataset', 'train', f"subject_train.txt")) - 1).unsqueeze(1).float()
    combined_tr = torch.cat((train_data[:,:-1], discrete_tr), 1)

    # select k best features on test data
    test_data = torch.tensor(selection.transform(test_x))
    discrete_te = torch.from_numpy(np.loadtxt(os.path.join('../dataset/UCI_HAR_Dataset', 'test', f"subject_test.txt")) - 1).unsqueeze(1).float()
    combined_te = torch.cat((test_data[:,:-1], discrete_te), 1)

    # generate training and testing data
    combined_data = torch.cat((combined_tr, combined_te))
    combined_label = torch.cat((train_y, test_y))

    # convert to dataframe
    combined_data = pd.DataFrame(combined_data)
    combined_data.columns = combined_data.columns.astype(str)
    combined_label = pd.DataFrame(combined_label)
    combined_label.columns = combined_label.columns.astype(str)

    # create training/testing test set
    combined_tr, combined_te, train_y, test_y = train_test_split(combined_data, combined_label, test_size=TE_TR_RATIO)

    ################
    ### TRAINING ###
    ################

    # makes ctgan model for each class for the Classifier
    models = []

    training_data, training_label = torch.from_numpy(combined_tr.values), torch.from_numpy(train_y.values).flatten()

    for i in range(0, len(classes)):
        # get training data for each class
        dataset_tr,_ = utils.get_activity_data(training_data, training_label, i)
        dataset_tr = pd.DataFrame(dataset_tr)
        dataset_tr.columns = dataset_tr.columns.astype(str)

        # names of the columns that are discrete
        discrete_columns = ['14']

        # initialize ctgan model
        model = CTGAN(epochs=EPOCHS)

        # train ctgan
        print("Starting training for class " + str(classes[i]))
        model.fit(dataset_tr, discrete_columns)

        # save models
        models.append(model)

    ##################
    ### EVALUATION ###
    ##################
    # get real data set to be the same size as generated data set
    combined_te = torch.from_numpy(combined_te[:NUM_SAMPLE*len(classes)].values)
    test_y = torch.from_numpy(test_y[:NUM_SAMPLE*len(classes)].values).flatten()

    # get data for each class
    ctgan_data = []
    ctgan_labels = []

    for i in range(0, len(classes)):
        # load trained ctgan model
        model = models[i]

        # get outputs of both models
        print("Generating data for " + str(classes[i]))
        synthetic_data = model.sample(NUM_SAMPLE)
        output = torch.from_numpy(synthetic_data.values)

        # add model outputs and labels to lists
        ctgan_data.append(output)
        ctgan_labels.append(torch.mul(torch.ones(NUM_SAMPLE), i))

    # concatenate data into single tensor
    ctgan_data, ctgan_labels = torch.cat(ctgan_data), torch.cat(ctgan_labels)

    # show PCA for all classes
    eval.recursive_pca_with_classes(combined_te, test_y, ctgan_data, ctgan_labels, classes, 100)

    # show PCA for each class
    for i in range(0, len(classes)):
        true_batch,_ = utils.get_activity_data(combined_te, test_y, i)
        fake_batch,_ = utils.get_activity_data(ctgan_data, ctgan_labels, i)
        eval.recursive_pca(true_batch, fake_batch, 100, title=f'{classes[i]}')
    plt.show()

    # machine evaluation for ctgan and GAN data
    print('Testing data from ctgan model')
    eval.binary_machine_evaluation(combined_te, test_y, ctgan_data, ctgan_labels, classes, TE_TR_RATIO, num_steps=NUM_STEPS)
    eval.multiclass_machine_evaluation(combined_te, test_y, ctgan_data, ctgan_labels, TE_TR_RATIO)
    eval.separability(combined_te, ctgan_data, TE_TR_RATIO)


if __name__ == '__main__':
    main()

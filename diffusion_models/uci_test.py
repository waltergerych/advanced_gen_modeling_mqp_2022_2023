# Code from https://github.com/azad-academy/denoising-diffusion-model/blob/main/diffusion_model_demo.ipynb
# Internal libraries
import diffusion as dfn
import evaluate as eval
import utils
from gan import generate_data, Generator
from helper_plot import hdr_plot_style
from model import ConditionalTabularModel

# External libraries
import matplotlib.pyplot as plt
import torch
from sklearn.feature_selection import SelectKBest


def main():
    # Set plot style
    hdr_plot_style()

    # load the datasets
    train_x, train_y = utils.load_data('../dataset/UCI_HAR_Dataset', 'train')
    test_x, test_y = utils.load_data('../dataset/UCI_HAR_Dataset', 'test')

    classes = ['WALKING', 'U-STAIRS', 'D-STAIRS', 'SITTING', 'STANDING', 'LAYING']

    labels = train_y
    dataset = train_x

    # Define the number of features from the dataset to use. Must be 561 or less
    NUM_FEATURES = 40
    NUM_STEPS = 1000
    NUM_REVERSE_STEPS = 1000
    LEARNING_RATE = .001
    BATCH_SIZE = 256
    HIDDEN_SIZE = 128
    NUM_SAMPLE = 1000

    # Use feature selection to select most important features
    feature_selector = SelectKBest(k=NUM_FEATURES)
    selection = feature_selector.fit(dataset, labels)
    features = selection.transform(dataset)
    dataset = torch.tensor(features)

    ################
    ### TRAINING ###
    ################

    # Makes diffusion model for each class for the Classifier
    models = []
    diffusions = []

    original_data, original_labels = dataset, labels

    for i in range(len(classes)):
        dataset, labels = utils.get_activity_data(original_data, original_labels, i)
        discrete = torch.where(dataset[:,0] > 0.25, 1, 0).unsqueeze(1)
        continuous = dataset[:,1:]

        diffusion = dfn.get_denoising_variables(NUM_STEPS)

        # extract number of discrete features and number of classes in each discrete feature
        feature_indices = []
        k = 0
        for j in range(discrete.shape[1]):
            # num = utils.get_classes(discrete[:, j]).shape[0]
            num=2
            feature_indices.append((k, k + num))
            k += num

        print("Starting training for class " + str(classes[i]))
        # print(NUM_STEPS, HIDDEN_SIZE, continuous.shape[1], k)
        try:
            model = ConditionalTabularModel(NUM_STEPS, HIDDEN_SIZE, continuous.shape[1], k)
            model.load_state_dict(torch.load(f'./diffusion_models/tabular_{classes[i]}_best{NUM_FEATURES}_{NUM_STEPS}.pth'))
            # model,_,_ = dfn.reverse_tabular_diffusion(discrete, continuous, diffusion, k, feature_indices, BATCH_SIZE, LEARNING_RATE, NUM_REVERSE_STEPS, model=model)
        except:
            model = ConditionalTabularModel(NUM_STEPS, HIDDEN_SIZE, continuous.shape[1], k)
            model,_,_ = dfn.reverse_tabular_diffusion(discrete, continuous, diffusion, k, feature_indices, BATCH_SIZE, LEARNING_RATE, NUM_REVERSE_STEPS, model=model)

        models.append(model)
        diffusions.append(diffusion)

        torch.save(model.state_dict(), f'./diffusion_models/tabular_{classes[i]}_best{NUM_FEATURES}_{NUM_STEPS}.pth')

    ##################
    ### EVALUATION ###
    ##################

    labels = test_y
    features = selection.transform(test_x)
    dataset = torch.tensor(features)

    test_train_ratio = .3

    # Get real data set to be the same size as generated data set
    dataset = dataset[:NUM_SAMPLE*len(classes)]
    labels = labels[:NUM_SAMPLE*len(classes)]
    discrete = torch.where(dataset[:,0] > 0.25, 1, 0).unsqueeze(1)
    continuous = dataset[:,1:]
    dataset = torch.cat((continuous, discrete), 1)

    # Get data for each class
    diffusion_data = []
    diffusion_labels = []

    # Get denoising variables
    ddpm = dfn.get_denoising_variables(NUM_STEPS)

    for i in range(len(classes)):
        # extract number of discrete features and number of classes in each discrete feature
        feature_indices = []
        k = 0
        for j in range(discrete.shape[1]):
            # num = utils.get_classes(discrete[:, j]).shape[0]
            num=2
            feature_indices.append((k, k + num))
            k += num

        # Load trained diffusion model
        # print(NUM_STEPS, HIDDEN_SIZE, continuous.shape[1], k)
        model = ConditionalTabularModel(NUM_STEPS, HIDDEN_SIZE, continuous.shape[1], k)
        model.load_state_dict(torch.load(f'./diffusion_models/tabular_{classes[i]}_best{NUM_FEATURES}_{NUM_STEPS}.pth'))

        # Get outputs of both models
        continuous_output, discrete_output,_ = utils.get_tabular_model_output(model, k, NUM_SAMPLE, feature_indices, continuous.shape[1], ddpm, calculate_continuous=True)

        # Add model outputs and labels to lists
        diffusion_data.append(torch.cat((continuous_output, discrete_output), 1))
        diffusion_labels.append(torch.mul(torch.ones(NUM_SAMPLE), i))
        print("Generated data for " + str(classes[i]))

    # Concatenate data into single tensor
    diffusion_data, diffusion_labels = torch.cat(diffusion_data), torch.cat(diffusion_labels)

    print(diffusion_data[:,0:2])
    print(dataset[:,0:2])
    print(diffusion_data[:,-1])
    print(dataset[:,-1])

    # Do PCA analysis for fake/real and subclasses
    eval.pca_with_classes(dataset, labels, diffusion_data, diffusion_labels, classes, overlay_heatmap=True)

    # Show PCA for each class
    # for i in range(len(classes)):
    #     true_batch,_ = utils.get_activity_data(dataset, labels, i)
    #     fake_batch,_ = utils.get_activity_data(diffusion_data, diffusion_labels, i)
    #     eval.perform_pca(true_batch, fake_batch, f'{classes[i]}')
    # plt.show()

    # Machine evaluation for diffusion and GAN data
    print('Testing data from diffusion model')
    eval.binary_machine_evaluation(dataset, labels, diffusion_data, diffusion_labels, classes, test_train_ratio, num_steps=NUM_STEPS)
    eval.multiclass_machine_evaluation(dataset, labels, diffusion_data, diffusion_labels, test_train_ratio)
    eval.separability(dataset, diffusion_data, test_train_ratio)


if __name__ == "__main__":
    main()

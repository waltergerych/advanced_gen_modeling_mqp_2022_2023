# Code from https://github.com/azad-academy/denoising-diffusion-model/blob/main/diffusion_model_demo.ipynb
# Internal libraries
import diffusion as dfn
import evaluate as eval
import utils
from helper_plot import hdr_plot_style
from model import ConditionalTabularModel

# External libraries
import matplotlib.pyplot as plt
import torch
from sklearn.feature_selection import SelectKBest


def main():
    # set plot style
    hdr_plot_style()

    # load the datasets
    train_x, train_y = utils.load_data('../dataset/UCI_HAR_Dataset', 'train')
    test_x, test_y = utils.load_data('../dataset/UCI_HAR_Dataset', 'test')

    classes = ['WALKING', 'U-STAIRS', 'D-STAIRS', 'SITTING', 'STANDING', 'LAYING']

    # define the number of features from the dataset to use. Must be 561 or less
    NUM_FEATURES = 10
    NUM_STEPS = 1000
    NUM_REVERSE_STEPS = 20000
    BATCH_SIZE = 256
    OPTIM_LR = .001
    CONTINUOUS_LR = 10
    HIDDEN_SIZE = 128
    NUM_SAMPLE = 1000
    TE_TR_RATIO = .3

    # use feature selection to select most important features
    feature_selector = SelectKBest(k=NUM_FEATURES)
    selection = feature_selector.fit(train_x, train_y)

    # select k best features on train data
    train_data = torch.tensor(selection.transform(train_x))
    discrete_tr = torch.where(train_data[:,0] > 0.25, 1, 0).unsqueeze(1)
    continuous_tr = train_data[:,1:]
    combined_tr = torch.cat((continuous_tr, discrete_tr), 1)

    # select k best features on test data
    test_data = torch.tensor(selection.transform(test_x))
    discrete_te = torch.where(test_data[:,0] > 0.25, 1, 0).unsqueeze(1)
    continuous_te = test_data[:,1:]
    combined_te = torch.cat((continuous_te, discrete_te), 1)

    # extract number of discrete features and number of classes in each discrete feature
    combined_discrete = torch.cat((discrete_tr, discrete_te), 0)
    feature_indices = []
    k = 0
    for j in range(combined_discrete.shape[1]):
        num = utils.get_classes(combined_discrete[:, j]).shape[0]
        feature_indices.append((k, k + num))
        k += num

    ################
    ### TRAINING ###
    ################

    # makes diffusion model for each class for the Classifier
    models = []

    original_data, original_labels = combined_tr, train_y

    for i in range(len(classes)):
        dataset,_ = utils.get_activity_data(original_data, original_labels, i)
        discrete = dataset[:,-1].unsqueeze(1)
        continuous = dataset[:,:-1]

        # initialize forward diffusion
        diffusion = dfn.get_denoising_variables(NUM_STEPS)

        print("Starting training for class " + str(classes[i]))
        try:
            model = ConditionalTabularModel(NUM_STEPS, HIDDEN_SIZE, continuous.shape[1], k)
            model.load_state_dict(torch.load(f'./diffusion_models/tabular_{classes[i]}_best{NUM_FEATURES}_{NUM_REVERSE_STEPS}.pth'))
            model,_,_ = dfn.reverse_tabular_diffusion(discrete, continuous, diffusion, k, feature_indices, BATCH_SIZE, OPTIM_LR, CONTINUOUS_LR, NUM_REVERSE_STEPS, model=model)
        except:
            model = ConditionalTabularModel(NUM_STEPS, HIDDEN_SIZE, continuous.shape[1], k)
            model,_,_ = dfn.reverse_tabular_diffusion(discrete, continuous, diffusion, k, feature_indices, BATCH_SIZE, OPTIM_LR, CONTINUOUS_LR, NUM_REVERSE_STEPS, model=model)

        # save models
        models.append(model)
        torch.save(model.state_dict(), f'./diffusion_models/tabular_{classes[i]}_best{NUM_FEATURES}_{NUM_REVERSE_STEPS}.pth')

    ##################
    ### EVALUATION ###
    ##################
    # get real data set to be the same size as generated data set
    combined_te = combined_te[:NUM_SAMPLE*len(classes)]
    test_y = test_y[:NUM_SAMPLE*len(classes)]

    # get data for each class
    diffusion_data = []
    diffusion_labels = []

    # get denoising variables
    ddpm = dfn.get_denoising_variables(NUM_STEPS)

    for i in range(len(classes)):
        # load trained diffusion model
        try:
            model = ConditionalTabularModel(NUM_STEPS, HIDDEN_SIZE, continuous_te.shape[1], k)
            model.load_state_dict(torch.load(f'./diffusion_models/tabular_{classes[i]}_best{NUM_FEATURES}_{NUM_REVERSE_STEPS}.pth'))
        except:
            model = models[i]

        # get outputs of both models
        continuous_output, discrete_output,_ = utils.get_tabular_model_output(model, k, NUM_SAMPLE, feature_indices, continuous_te.shape[1], ddpm, calculate_continuous=True)

        # add model outputs and labels to lists
        diffusion_data.append(torch.cat((continuous_output, discrete_output), 1))
        diffusion_labels.append(torch.mul(torch.ones(NUM_SAMPLE), i))
        print("Generated data for " + str(classes[i]))

    # concatenate data into single tensor
    diffusion_data, diffusion_labels = torch.cat(diffusion_data), torch.cat(diffusion_labels)

    # do PCA analysis for fake/real and subclasses
    eval.pca_with_classes(combined_te, test_y, diffusion_data, diffusion_labels, classes, overlay_heatmap=True)

    # show PCA for each class
    for i in range(len(classes)):
        true_batch,_ = utils.get_activity_data(combined_te, test_y, i)
        fake_batch,_ = utils.get_activity_data(diffusion_data, diffusion_labels, i)
        eval.perform_pca(true_batch, fake_batch, f'{classes[i]}')
    plt.show()

    # machine evaluation for diffusion and GAN data
    print('Testing data from diffusion model')
    eval.binary_machine_evaluation(combined_te, test_y, diffusion_data, diffusion_labels, classes, TE_TR_RATIO, num_steps=NUM_STEPS)
    eval.multiclass_machine_evaluation(combined_te, test_y, diffusion_data, diffusion_labels, TE_TR_RATIO)
    eval.separability(combined_te, diffusion_data, TE_TR_RATIO)


if __name__ == "__main__":
    main()

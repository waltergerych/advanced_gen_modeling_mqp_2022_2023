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
from sklearn.model_selection import train_test_split


def main():
    # set plot style
    hdr_plot_style()

    # set training
    set_train = True

    # load the datasets
    train_x, train_y = utils.load_data('../dataset/UCI_HAR_Dataset', 'train')
    test_x, test_y = utils.load_data('../dataset/UCI_HAR_Dataset', 'test')

    classes = ['WALKING', 'U-STAIRS', 'D-STAIRS', 'SITTING', 'STANDING', 'LAYING']

    # define the number of features from the dataset to use. Must be 561 or less
    NUM_FEATURES = 15
    NUM_STEPS = 10000
    NUM_REVERSE_STEPS = 10000
    BATCH_SIZE = 128
    OPTIM_LR = .001
    CONTINUOUS_LR = 3
    HIDDEN_SIZE = 128
    VALIDATION_RATIO = .2
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

    # create validation test set
    combined_te, combined_vl, test_y, validation_y = train_test_split(combined_te, test_y, test_size=VALIDATION_RATIO)

    ################
    ### TRAINING ###
    ################

    # makes diffusion model for each class for the Classifier
    models = []
    training_loss_list = []
    validation_loss_list = []
    discrete_probs_list = []

    training_data, training_label = combined_tr, train_y
    validation_data, validation_label = combined_vl, validation_y

    if set_train:
        for i in range(0, len(classes)):
            # get training data for each class
            dataset_tr,_ = utils.get_activity_data(training_data, training_label, i)
            discrete_tr = dataset_tr[:,-1].unsqueeze(1)
            continuous_tr = dataset_tr[:,:-1]

            # get validation data for each class
            dataset_vl,_ = utils.get_activity_data(validation_data, validation_label, i)
            discrete_vl = dataset_vl[:,-1].unsqueeze(1)
            continuous_vl = dataset_vl[:,:-1]

            # initialize forward diffusion
            diffusion = dfn.get_denoising_variables(NUM_STEPS)

            print("Starting training for class " + str(classes[i]))
            try:
                model = ConditionalTabularModel(NUM_STEPS, HIDDEN_SIZE, continuous_tr.shape[1], k)
                model.load_state_dict(torch.load(f'./diffusion_models/tabular_{classes[i]}_best{NUM_FEATURES}_forward{NUM_STEPS}_reverse{NUM_REVERSE_STEPS}.pth'))
                model, class_training_loss, class_validation_loss, probs = dfn.reverse_tabular_diffusion(discrete_tr,
                                                                                                         continuous_tr,
                                                                                                         discrete_vl,
                                                                                                         continuous_vl,
                                                                                                         diffusion,
                                                                                                         k,
                                                                                                         feature_indices,
                                                                                                         BATCH_SIZE,
                                                                                                         OPTIM_LR,
                                                                                                         CONTINUOUS_LR,
                                                                                                         NUM_REVERSE_STEPS,
                                                                                                         model=model,
                                                                                                         show_loss=True)
            except:
                model = ConditionalTabularModel(NUM_STEPS, HIDDEN_SIZE, continuous_tr.shape[1], k)
                model, class_training_loss, class_validation_loss, probs = dfn.reverse_tabular_diffusion(discrete_tr,
                                                                                                         continuous_tr,
                                                                                                         discrete_vl,
                                                                                                         continuous_vl,
                                                                                                         diffusion,
                                                                                                         k,
                                                                                                         feature_indices,
                                                                                                         BATCH_SIZE,
                                                                                                         OPTIM_LR,
                                                                                                         CONTINUOUS_LR,
                                                                                                         NUM_REVERSE_STEPS,
                                                                                                         model=model,
                                                                                                         show_loss=True)

            # capture statistics
            training_loss_list.append(class_training_loss)
            validation_loss_list.append(class_validation_loss)
            discrete_probs_list.append(probs)

            # save models
            models.append(model)
            torch.save(model.state_dict(), f'./diffusion_models/tabular_{classes[i]}_best{NUM_FEATURES}_forward{NUM_STEPS}_reverse{NUM_REVERSE_STEPS}.pth')

        # show each classes training/validation loss
        for i in range(len(classes)):
            eval.plot_loss_and_discrete_distribution(f'{classes[i]}', training_loss_list[i], validation_loss_list[i], discrete_probs_list[i])
        plt.show()

    ##################
    ### EVALUATION ###
    ##################
    # get real data set to be the same size as generated data set
    combined_te = combined_te[:NUM_SAMPLE*len(classes)]
    test_y = test_y[:NUM_SAMPLE*len(classes)]
    discrete_te = combined_te[:,-1].unsqueeze(1)
    continuous_te = combined_te[:,:-1]

    # get data for each class
    diffusion_data = []
    diffusion_labels = []

    # get denoising variables
    ddpm = dfn.get_denoising_variables(NUM_STEPS)

    for i in range(len(classes)):
        # load trained diffusion model
        try:
            model = ConditionalTabularModel(NUM_STEPS, HIDDEN_SIZE, continuous_te.shape[1], k)
            model.load_state_dict(torch.load(f'./diffusion_models/tabular_{classes[i]}_best{NUM_FEATURES}_forward{NUM_STEPS}_reverse{NUM_REVERSE_STEPS}.pth'))
        except:
            model = models[i]

        # get outputs of both models
        print("Generating data for " + str(classes[i]))
        continuous_output, discrete_output,_ = utils.get_tabular_model_output(model, k, NUM_SAMPLE, feature_indices, continuous_te.shape[1], ddpm, calculate_continuous=True)

        # add model outputs and labels to lists
        diffusion_data.append(torch.cat((continuous_output, discrete_output), 1))
        diffusion_labels.append(torch.mul(torch.ones(NUM_SAMPLE), i))

    # concatenate data into single tensor
    diffusion_data, diffusion_labels = torch.cat(diffusion_data), torch.cat(diffusion_labels)

    # show PCA for all classes
    eval.recursive_pca_with_classes(combined_te, test_y, diffusion_data, diffusion_labels, classes, 100)

    # show PCA for each class
    for i in range(len(classes)):
        true_batch,_ = utils.get_activity_data(combined_te, test_y, i)
        fake_batch,_ = utils.get_activity_data(diffusion_data, diffusion_labels, i)
        eval.recursive_pca(true_batch, fake_batch, 100, title=f'{classes[i]}')
    plt.show()

    # machine evaluation for diffusion and GAN data
    print('Testing data from diffusion model')
    eval.binary_machine_evaluation(combined_te, test_y, diffusion_data, diffusion_labels, classes, TE_TR_RATIO, num_steps=NUM_STEPS)
    eval.multiclass_machine_evaluation(combined_te, test_y, diffusion_data, diffusion_labels, TE_TR_RATIO)
    eval.separability(combined_te, diffusion_data, TE_TR_RATIO)


if __name__ == "__main__":
    main()

# Code from https://github.com/azad-academy/denoising-diffusion-model/blob/main/diffusion_model_demo.ipynb
# Internal libraries
import diffusion as dfn
import evaluate as eval
import utils
from model import ConditionalTabularModel

# External libraries
import matplotlib.pyplot as plt
import torch
from sklearn.feature_selection import SelectKBest


def main():
    # load the datasets
    train_x, train_y = utils.load_data('../dataset/UCI_HAR_Dataset', 'train')
    test_x, test_y = utils.load_data('../dataset/UCI_HAR_Dataset', 'test')

    # classes = [0, 1, 2, 3, 4, 5]
    classes = ['WALKING', 'U-STAIRS', 'D-STAIRS', 'SITTING', 'STANDING', 'LAYING']

    dataset, labels = train_x, train_y

    # declare hyperparameters
    NUM_SAMPLES = 986
    NUM_STEPS = 1000
    NUM_REVERSE_STEPS = 100
    LEARNING_RATE = .001
    BATCH_SIZE = 128
    HIDDEN_SIZE = 128

    # use feature selection to select most important features
    feature_selector = SelectKBest(k=40)
    selection = feature_selector.fit(dataset, labels)
    features = selection.transform(dataset)
    dataset = torch.tensor(features)

    #################
    ### DIFFUSION ###
    #################

    # makes diffusion model for each class for the Classifier
    models = []
    diffusions = []

    original_data, original_labels = dataset, labels

    for i in range(len(classes)):
        continuous, _ = utils.get_activity_data(original_data, original_labels, i)
        continuous = continuous[:NUM_SAMPLES]
        # dummy discrete
        test_discrete = []
        w1 = torch.tensor([.95, .05])
        w2 = torch.tensor([.1, .3, .6])
        test_discrete.append(torch.multinomial(w1, NUM_SAMPLES, replacement=True))
        test_discrete.append(torch.multinomial(w2, NUM_SAMPLES, replacement=True))
        discrete = torch.stack(test_discrete, dim=1)

        # forward diffusion
        diffusion = dfn.get_denoising_variables(NUM_STEPS)

        # extract number of discrete features and number of classes in each discrete feature
        feature_indices = []
        k = 0
        for j in range(discrete.shape[1]):
            num = utils.get_classes(discrete[:, j]).shape[0]
            feature_indices.append((k, k + num))
            k += num

        # reverse diffusion
        print(f'Starting training for class {classes[i]}')
        model = ConditionalTabularModel(NUM_STEPS, HIDDEN_SIZE, continuous.shape[1], k)
        # model.load_state_dict(torch.load(f'./models/tabular_{NUM_STEPS}_{classes[i]}.pth'))
        model, loss, probs = dfn.reverse_tabular_diffusion(discrete, continuous, diffusion, k, feature_indices, BATCH_SIZE, LEARNING_RATE, NUM_REVERSE_STEPS, model=model)

        # save model
        models.append(model)
        diffusions.append(diffusion)
        # torch.save(model.state_dict(), f'./models/tabular_{NUM_STEPS}_{classes[i]}.pth')

    ##################
    ### EVALUATION ###
    ##################

    labels = test_y
    features = selection.transform(test_x)
    dataset = torch.tensor(features)

    input_size = dataset.shape[1]
    num_to_gen = 150
    test_train_ratio = .3

    # Gan variables
    generator_input_size = 128
    hidden_size = 512

    # Get real data set to be the same size as generated data set
    dataset = dataset[:num_to_gen*len(classes)]
    labels = labels[:num_to_gen*len(classes)]

    # Get data for each class
    diffusion_data = []
    diffusion_labels = []
    gan_data = []
    gan_labels = []

    # Get denoising variables
    ddpm = dfn.get_denoising_variables(NUM_STEPS)

    for i in range(len(classes)):
        # Load trained diffusion model
        model = ConditionalModel(NUM_STEPS, dataset.size(1))
        model.load_state_dict(torch.load(
            f'./models/{NUM_STEPS}_step_model_best40_{i}.pth'))

        # Load trained GAN model
        generator = Generator(generator_input_size,
                              hidden_size, dataset.size(1))
        generator.load_state_dict(torch.load(
            f'./gan/generator/G_{classes[i]}.pth'))

        # Get outputs of both models
        diffusion_output = utils.get_model_output(
            model, input_size, ddpm, num_to_gen)
        gan_output = generate_data(
            [generator], num_to_gen, generator_input_size)

        # CODE TO GRAPH 10 PLOTS OF REMOVING NOISE FOR EACH CLASS
        # --> MUST CHANGE 'get_model_output' to return x_seq rather than x_seq[-1]
        # true_batch, true_labels = get_activity_data(dataset, labels, i)
        # for j in range(0, NUM_STEPS, 10):
        #     perform_pca(true_batch, output[j], f'T{100-j}')
        # perform_pca(true_batch, output[-1], 'T0')
        # plt.show()

        # Add model outputs and labels to lists
        diffusion_data.append(diffusion_output)
        diffusion_labels.append(torch.mul(torch.ones(num_to_gen), i))
        gan_data.append(gan_output[0])
        gan_labels.append(torch.mul(torch.ones(num_to_gen), i))
        print("Generated data for " + str(classes[i]))

    # Concatenate data into single tensor
    diffusion_data, diffusion_labels = torch.cat(
        diffusion_data), torch.cat(diffusion_labels)
    gan_data, gan_labels = torch.cat(gan_data), torch.cat(gan_labels)

    # Do PCA analysis for fake/real and subclasses
    eval.pca_with_classes(dataset, labels, diffusion_data,
                          diffusion_labels, classes, overlay_heatmap=True)

    # Show PCA for each class
    for i in range(len(classes)):
        true_batch, _ = utils.get_activity_data(dataset, labels, i)
        fake_batch, _ = utils.get_activity_data(
            diffusion_data, diffusion_labels, i)
        eval.perform_pca(true_batch, fake_batch, f'{classes[i]}')
    plt.show()

    # Machine evaluation for diffusion and GAN data
    print('Testing data from diffusion model')
    eval.binary_machine_evaluation(dataset, labels, diffusion_data,
                                   diffusion_labels, classes, test_train_ratio, num_steps=NUM_STEPS)
    eval.multiclass_machine_evaluation(
        dataset, labels, diffusion_data, diffusion_labels, test_train_ratio)
    eval.separability(dataset, diffusion_data, test_train_ratio)

    # print('Testing data from gan model')
    # binary_machine_evaluation(dataset, labels, gan_data, gan_labels, classes, test_train_ratio)
    # multiclass_machine_evaluation(dataset, labels, gan_data, gan_labels, test_train_ratio)
    # separability(dataset, gan_data, test_train_ratio)


if __name__ == "__main__":
    main()

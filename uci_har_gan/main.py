# Native libraries
import sys
# Internal libraries
import utils
import gan
import classifier
import visualize
# External libraries
import torch.optim as optim
import torch.nn as nn
import torch


def main():
    """Main function"""
    # If set to true, new models will be created and trained. Otherwise, a model will be loaded and evaluated
    training = False

    # load the datasets
    train_x, train_y = utils.load_data('../dataset/UCI_HAR_Dataset', 'train')
    test_x, test_y = utils.load_data('../dataset/UCI_HAR_Dataset', 'test')

    classes = [0, 1, 2, 3, 4, 5]

    # initialize hyperparameters
    input_size = 128
    hidden_size = 512
    epoch = 5000
    batch_size = 100
    learning_rate = 0.005
    momentum = 0.9
    train_ratio = 5

    # models
    generators = []
    discriminators = []

    # optimizers
    generator_optimizers = []
    discriminator_optimizers = []

    if training:
        for i in classes:
            generators.append(gan.Generator(input_size, hidden_size, train_x.size(1)))
            discriminators.append(gan.Discriminator(train_x.size(1), hidden_size))
            generator_optimizers.append(optim.SGD(generators[i].parameters(), lr=learning_rate, momentum=momentum))
            discriminator_optimizers.append(optim.SGD(discriminators[i].parameters(), lr=learning_rate, momentum=momentum))

            # loss function
            criterion = nn.BCELoss()

            # retrieve data for specified class
            x, y = utils.get_activity_data(train_x, train_y, i)

            # train the models
            gan.train_model(generators[i], discriminators[i], generator_optimizers[i], discriminator_optimizers[i], criterion, x, y, epoch, batch_size, input_size, train_ratio)

            # place in eval mode
            generators[i].eval()

            # save the model
            torch.save(generators[i].state_dict(), f"./generator/G{i}_model.pth")
            torch.save(discriminators[i].state_dict(), f"./discriminator/D{i}_model.pth")
    else:
        generators = []
        for i in classes:
            generator = gan.Generator(input_size, hidden_size, train_x.size(1))
            generator.load_state_dict(torch.load(f'./generator/G{i}_model.pth'))
            generators.append(generator)

    # test the model
    test_data = test_x, test_y

    test_size = 500
    train_size = 1000

    # train a new classifier with the given generators
    # new_classifier = classifier.train_classifier(generators, train_size, input_size)
    # torch.save(new_classifier.state_dict(), 'fake_trained_classifier.pth')

    # evaluate with classifiers
    print("\nClassifying generated data using a classifier pretrained on real data")
    classifier.evaluate(generators, test_size, input_size, test_data, 'real_trained_classifier.pth')
    # print("\nClassifying generated data using a classifier pretrained on fake data")
    # classifier.evaluate(generators, test_size, input_size, test_data, 'fake_trained_classifier.pth')

    # visualize with histograms (currently only visualizing the walking class)
    # data_x, data_y = gan.generate_data([generators[0]], test_size, input_size)
    # visualize.make_histograms(data_x)
    # visualize.divergence(test_x, data_x)

if __name__ == "__main__":
    main()

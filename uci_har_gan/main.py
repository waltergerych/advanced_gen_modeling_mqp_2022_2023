# Native libraries
import sys
import numpy as np
from sklearn.model_selection import train_test_split
# Internal libraries
import utils
import gan
import classifier
import visualize
# External libraries
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.linear_model import LogisticRegression
from regression import train_regression_model, generate_data_and_labels


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
    # print("\nClassifying generated data using a classifier pretrained on real data")
    # classifier.evaluate(generators, test_size, input_size, test_data, './classifier/real_trained_classifier.pth')
    # print("\nClassifying generated data using a classifier pretrained on fake data")
    # classifier.evaluate(generators, test_size, input_size, test_data, './classifier/fake_trained_classifier.pth')



    # train a logistic regression model to tell real and fake data apart
    # logistic_regression_model = LogisticRegression()
    # # generate fake data
    # fake_data, _ = gan.generate_data(generators, train_size, input_size)
    
    # # get data and labels for train test split
    # all_data, all_labels = generate_data_and_labels(fake_data, train_x)

    # # generate train test split (70/30 split)
    # X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.3, stratify=all_labels)

    # # get accuracy, precision, and recall of the model
    # accuracy, precision, recall = train_regression_model(logistic_regression_model, X_train, y_train, X_test, y_test)
    # print("Logistic Regression Model Stats:")
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    


    # visualize with histograms (currently only visualizing the walking class)
    # data_x, data_y = gan.generate_data([generators[0]], test_size, input_size)
    # visualize.make_histograms(data_x)

    # generate data from all classes
    data_x, data_y = gan.generate_data(generators, test_size, input_size)
    visualize.divergence(test_x, test_y, data_x, data_y)
    # visualize.perform_pca(test_x, data_x)

if __name__ == "__main__":
    main()

# Internal libraries
import utils
import model
import evaluate
# External libraries
import torch.optim as optim
import torch.nn as nn
import torch


def main():
    """Main function"""
    # load the datasets
    train_x, train_y = utils.load_data('../UCI_HAR_Dataset', 'train')
    test_x, test_y = utils.load_data('../UCI_HAR_Dataset', 'test')

    classes = [0, 1, 2, 3, 4, 5]

    # initialize hyperparameters
    hidden_size = 512
    epoch = 5
    batch_size = 100
    learning_rate = 0.005
    momentum = 0.5

    feature_size = 561

    # models
    generators = []
    discriminators = []

    # optimizers
    generator_optimizers = []
    discriminator_optimizers = []

    for i in classes:
        generators.append(model.Generator(train_x.size(1), hidden_size))
        discriminators.append(model.Discriminator(train_x.size(1), hidden_size))
        generator_optimizers.append(optim.SGD(generators[i].parameters(), lr=learning_rate, momentum=momentum))
        discriminator_optimizers.append(optim.SGD(discriminators[i].parameters(), lr=learning_rate, momentum=momentum))

        # loss function
        criterion = nn.BCELoss()

        # retrieve data for specified class
        x, y = utils.get_activity_data(train_x, train_y, i)

        # train the models
        model.train_model(generators[i], discriminators[i], generator_optimizers[i], discriminator_optimizers[i], criterion, x, y, epoch, batch_size)

        # place in eval mode
        generators[i].eval()

    # test the model
    generated_data_x = []
    generated_data_y = []
    for i in classes:
        noise = torch.randn(size=(batch_size*2, feature_size)).float()
        generated_data_x.append(generators[i](noise))
        generated_data_y.append(torch.mul(torch.ones(batch_size*2), i))

    combined_generated_data_x = torch.cat(generated_data_x)
    combined_generated_data = combined_generated_data_x, torch.cat(generated_data_y)
    true_data = test_x, test_y

    evaluate.evaluate(true_data, combined_generated_data, 'group_model_classifier.pth')

if __name__ == "__main__":
    main()

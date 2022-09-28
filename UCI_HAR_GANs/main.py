# Internal libraries
import utils
import gan
import classifier
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
    input_size = 128
    hidden_size = 512
    epoch = 5000
    batch_size = 100
    learning_rate = 0.005
    momentum = 0.5
    train_ratio = 5

    # models
    generators = []
    discriminators = []

    # optimizers
    generator_optimizers = []
    discriminator_optimizers = []

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
        torch.save(generators[i].state_dict(), f"./G{i}_model.pth")
        torch.save(discriminators[i].state_dict(), f"./D{i}_model.pth")

    # test the model
    generated_data_x = []
    generated_data_y = []
    for i in classes:
        noise = torch.randn(size=(batch_size*2, input_size)).float()
        generated_data_x.append(generators[i](noise))
        generated_data_y.append(torch.mul(torch.ones(batch_size*2), i))

    combined_generated_data_x = torch.cat(generated_data_x)
    combined_generated_data = combined_generated_data_x, torch.cat(generated_data_y)
    true_data = test_x, test_y

    classifier.evaluate(true_data, combined_generated_data, 'group_model_classifier.pth')

if __name__ == "__main__":
    main()

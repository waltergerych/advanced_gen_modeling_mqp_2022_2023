# Internal libraries
import utils
import model
# External libraries
import torch.optim as optim
import torch.nn as nn


def main():
    """Main function"""
    # load the datasets
    train_x, train_y = utils.load_data('../UCI_HAR_Dataset', 'train')
    test_x, test_y = utils.load_data('../UCI_HAR_Dataset', 'test')

    # get the data for each activity labels
    walking_x, walking_y = utils.get_activity_data(train_x, train_y, 0)
    upstairs_x, upstairs_y = utils.get_activity_data(train_x, train_y, 1)
    downstairs_x, downstairs_y = utils.get_activity_data(train_x, train_y, 2)
    sitting_x, sitting_y = utils.get_activity_data(train_x, train_y, 3)
    standing_x, standing_y = utils.get_activity_data(train_x, train_y, 4)
    laying_x, laying_y = utils.get_activity_data(train_x, train_y, 5)

    # initialize hyperparameters
    hidden_size = 512
    epoch = 5
    batch_size = 100
    learning_rate = 0.005
    momentum = 0.5

    # models
    generator = model.Generator(train_x.size(1), hidden_size)
    discriminator = model.Discriminator(train_x.size(1), hidden_size)

    # optimizers
    generator_optimizer = optim.SGD(generator.parameters(), lr=learning_rate, momentum=momentum)
    discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=learning_rate, momentum=momentum)

    # loss function
    criterion = nn.BCELoss()

    # train the models
    model.train_model(generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, walking_x, walking_y, epoch, batch_size)


if __name__ == "__main__":
    main()

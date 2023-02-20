# External libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *

class Generator(nn.Module):
    """Class for generating fake HAR data"""

    def __init__(self, input_size, hidden_size, output_size):
        """Class constructor for generator.

        First layer converts input data into the specified hidden layer size.
        Second layer takes the output of the first layer through ReLU activation function.
        Third layer converts hidden layer into an output of the same size input from the dataset.
        Fourth layer takes the output through a tanh function for normalization.

        Args:
            input_size (torch.Tensor): the input size of the real data.
            hidden_size (int): the hidden layer size.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)


    def forward(self, data):
        """Forward propagation for the model.

        Args:
            data (torch.Tensor): the data that are passed through the network's layers.

        Returns:
            data (torch.Tensor): the output data from all the layers.
        """
        data = self.fc1(data)
        data = F.relu(data)
        data = self.output(data)
        data = torch.tanh(data)
        return data


class Discriminator(nn.Module):
    """Class for discriminating data as real/fake"""

    def __init__(self, input_size, hidden_size):
        """Class constructor for discriminator.

        First layer converts input data into the specified hidden layer size.
        Second layer takes the output of the first layer through ReLU activation function.
        Third layer converts hidden layer into an output between 0 and 1.
        Fourth layer takes the output through a sigmoid function for normalization.

        Args:
            input_size (torch.Tensor): the input size of the real data.
            hidden_size (int): the hidden layer size.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)


    def forward(self, data):
        """Forward propagation for the discriminator.

        Args:
            data (torch.Tensor): the data that are passed through the network's layers.

        Returns:
            data (torch.Tensor): the output data from all the layers.
        """
        data = self.fc1(data)
        data = F.relu(data)
        data = self.output(data)
        data = torch.sigmoid(data)
        return data


def train_model(generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, true_x, true_y, epoch, batch_size, input_size, ratio):
    """Train the generator and discriminator with the specified hyperparameters.

    Args:
        generator (Generator): the generator network.
        discriminator (Discriminator): the discriminator network.
        generator_optimizer (torch.optim): the generator optimizer.
        discriminator_optimizer (torch.optim): the discriminator optimizer.
        criterion (torch.nn.modules.loss): loss function for the models.
        true_x (torch.Tensor): real data features.
        true_y (torch.Tensor): real data labels.
        epoch (int): number of training epochs.
        batch_size (int): size of training batch.
        ratio (int): the training ratio between generator and discriminator
    """
    # initialize the state of the discriminator and generator at the beginning
    discriminator.eval()
    generator.train()

    # loop through the dataset multiple times
    for e in range(epoch):
        # print epoch for progess update
        if e % 500 == 0:
            print(e)
        # randomize the samples
        rand_idx = np.random.permutation(true_x.size(0))
        rand_x, rand_y = true_x[rand_idx,:], true_y[rand_idx]

        # process the epoch batch by batch
        for i in range(0, (true_y.size(0)//batch_size)):
            # initialize the starting and ending index of the current batch
            start_idx = i*batch_size
            end_idx = start_idx+batch_size
            batch_x = rand_x[start_idx:end_idx,:]
            batch_y = rand_y[start_idx:end_idx]

            # generate a noise data from normal distribution
            noise = torch.randn(size=(batch_size, input_size)).float()
            generated_data = generator(noise)

            # train the generator
            generator_optimizer.zero_grad()
            generator_discriminator_out = discriminator(generated_data)
            generator_loss = criterion(generator_discriminator_out.flatten(), batch_y)
            generator_loss.backward()
            generator_optimizer.step()

            if e % ratio == 0:
                # unfreeze discriminator and freeze generator
                discriminator.train()
                generator.eval()

                # train the discriminator on the true data
                discriminator_optimizer.zero_grad()
                true_discriminator_out = discriminator(batch_x)
                true_discriminator_loss = criterion(true_discriminator_out.flatten(), batch_y)

                # train the discriminator on the fake data
                generator_discriminator_out = discriminator(generated_data.detach())
                generator_discriminator_loss = criterion(generator_discriminator_out.flatten(), torch.zeros(batch_size))
                discriminator_loss = (true_discriminator_loss + generator_discriminator_loss)
                discriminator_loss.backward()
                discriminator_optimizer.step()

                # freeze the discriminator and unfreeze generator
                discriminator.eval()
                generator.train()

def generate_data(generators, batch_size, input_size):
    """ Generates data using the given generators

    Args:
        generators (List<Generator>): the list of generators to be used to make the data
        batch_size (int): the number of data instances to be created for each generator
        input_size (int): the size of the noise to be used as input for the generator

    Returns:
        generated_x (torch.Tensor): the generated data in the x domain
        generated_y (torch.Tensor): the generated label for the data
    """
    generated_data_x = []
    generated_data_y = []
    for i, generator in enumerate(generators):
        noise = torch.randn(size=(batch_size, input_size)).float()
        generated_data_x.append(generator(noise))
        generated_data_y.append(torch.mul(torch.ones(batch_size), i))

    generated_x, generated_y = torch.cat(generated_data_x), torch.cat(generated_data_y)

    return generated_x, generated_y

def make_gans(train_x, train_y, classes):
    """Builds and trains gans for each of the six HAR classes

    Args:
        train_x (torch.Tensor): the training data
        train_y (torch.Tensor): the class labels for the training data
        classes (list<strings>): a list of the classes
    """

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

    for i in range(len(classes)):
        generators.append(Generator(input_size, hidden_size, train_x.size(1)))
        discriminators.append(Discriminator(train_x.size(1), hidden_size))
        generator_optimizers.append(optim.SGD(generators[i].parameters(), lr=learning_rate, momentum=momentum))
        discriminator_optimizers.append(optim.SGD(discriminators[i].parameters(), lr=learning_rate, momentum=momentum))

        generators[i].load_state_dict(torch.load(f"./gan/generator/G_{classes[i]}.pth"))
        discriminators[i].load_state_dict(torch.load(f"./gan/discriminator/D_{classes[i]}.pth"))

        generators[i].train()

        # loss function
        criterion = nn.BCELoss()

        # retrieve data for specified class
        x, y = get_activity_data(train_x, train_y, i)

        # train the models
        print(f'Training {classes[i]} GAN')
        train_model(generators[i], discriminators[i], generator_optimizers[i], discriminator_optimizers[i], criterion, x, y, epoch, batch_size, input_size, train_ratio)

        # place in eval mode
        generators[i].eval()

        # save the model
        torch.save(generators[i].state_dict(), f"./gan/generator/G_{classes[i]}.pth")
        torch.save(discriminators[i].state_dict(), f"./gan/discriminator/D_{classes[i]}.pth")

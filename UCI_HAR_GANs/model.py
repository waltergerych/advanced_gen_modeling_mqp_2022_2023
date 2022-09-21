# External libraries
import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Class for generating fake HAR data
    """

    def __init__(self, input_size, hidden_size):
        """
        Class constructor for generator.

        First layer converts input data into the specified hidden layer size.
        Second layer takes the output of the first layer through ReLU activation function.
        Third layer converts hidden layer into an output of the same size input from the dataset.
        Fourth layer takes the output through a sigmoid function for normalization.

        @param: input_size: int, hidden_size: int
        @return: None
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, data):
        """
        Forward propagation for the model.

        @param: data: torch.Tensor
        @return: data: torch.Tensor
        """
        data = self.fc1(data)
        data = self.relu(data)
        data = self.output(data)
        data = self.sigmoid(data)
        return data


class Discriminator(nn.Module):
    """
    Class for discriminating data as real/fake
    """

    def __init__(self, input_size, hidden_size):
        """
        Class constructor for discriminator.

        First layer converts input data into the specified hidden layer size.
        Second layer takes the output of the first layer through ReLU activation function.
        Third layer converts hidden layer into an output of the same size input from the dataset.
        Fourth layer takes the output through a sigmoid function for normalization.

        @param: input_size: int, hidden_size: int
        @return: None
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, data):
        """
        Forward propagation for the model.

        @param: data: torch.Tensor
        @return: data: torch.Tensor
        """
        data = self.fc1(data)
        data = self.relu(data)
        data = self.output(data)
        data = self.sigmoid(data)
        return data


def train_model(generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, true_x, true_y, epoch, batch_size, show_loss=False):
    """
    Train the generator and discriminato with the specified hyperparameters

    @param: generator: Generator, discriminator: Discriminator,
            generator_optimizer: torch.optim, discriminator_optimizer: torch.optim,
            criterion: torch.nn.modules.loss, true_x: torch.Tensor, true_y: torch.Tensor,
            epoch: int, batch_size: int, show_loss: bool
    @return: None
    """

    # loop through the dataset multiple times
    for e in range(epoch):
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
            noise = torch.randn(size=(batch_size, batch_x.size(1))).float()
            generated_data = generator(noise)

            # train the generator
            generator_discriminator_out = discriminator(generated_data)
            generator_loss = criterion(generator_discriminator_out.flatten(), batch_y)
            generator_loss.backward()
            generator_optimizer.step()

            # train the discriminator on the true data
            discriminator_optimizer.zero_grad()
            true_discriminator_out = discriminator(batch_x)
            true_discriminator_loss = criterion(true_discriminator_out.flatten(), batch_y)

            # add .detach() here think about this
            generator_discriminator_out = discriminator(generated_data.detach())
            generator_discriminator_loss = criterion(generator_discriminator_out.flatten(), torch.zeros(batch_size))
            discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()

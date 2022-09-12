# Native libraries
import os
# External libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class FF(nn.Module):
    """
    Class for feed-forward model from pytorch.
    """

    def __init__(self, input_size, hidden_size):
        """
        Class constructor for feed-forward model.

        First layer converts input data into the specified hidden layer size.
        Second layer takes the output of the first layer through ReLU activation function.
        Third layer converts hidden layer into 6 classes from the dataset

        @param: input_size: int, hidden_size: int
        @return: None
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 6)


    def forward(self, data):
        """
        Forward propagation for the model.

        @param: data: torch.Tensor
        @return: data: torch.Tensor
        """
        data = self.fc1(data)
        data = self.relu(data)
        data = self.fc2(data)
        return data


def load_data(dataset, dataset_type):
    """
    Load data from a given dataset name and dataset type (train/test).
    The function expects the data to be in the following format:
    "{dataset}/{dataset_type}/(X|y)_{dataset_type}.txt"

    @param: dataset: string, dataset_type: string
    @return: data: torch.Tensor, labels: torch.Tensor
    """
    # load data and its labels
    x = np.loadtxt(os.path.join(dataset, dataset_type, f"X_{dataset_type}.txt"))
    y = np.loadtxt(os.path.join(dataset, dataset_type, f"y_{dataset_type}.txt"))

    # convert loaded data from numpy to tensor
    data = torch.from_numpy(x).float()
    labels = torch.from_numpy(y).long()

    # convert 1-indexed class labels to 0-indexed labels
    labels -= 1

    return data, labels


def fPC(model, data, labels):
    """
    Measure of percent correct of the current model

    @param: data: torch.Tensor, labels: torch.Tensor
    @return: None
    """
    with torch.no_grad():
        # run the data through the model
        outputs = model(data)
        # get the predicted result from the output of the model
        _, predicted = torch.max(outputs.data, 1)
        # get the number of correct guesses
        correct = (predicted == labels).sum().item()
        # calculate the percent correct
        percent_correct = correct / labels.size(0)

    return percent_correct


def train_model(model, optimizer, criterion, train_x, train_y, epochs, batch_size, show_loss=False):
    """
    Train the model with the specified hyperparameters

    @param: model: FF, optimizer: torch.optim, criterion: torch.nn.modules.loss
    @return: None
    """
    # loop through the dataset multiple times
    for e in range(epochs):
        # keep track of current loss for statistics
        curr_loss = 0.0

        # randomize the samples
        rand_idx = np.random.permutation(train_x.size(0))
        rand_x, rand_y = train_x[rand_idx,:], train_y[rand_idx]

        # process the epoch batch by batch
        for i in range(0, (train_y.size(0)//batch_size)):
            # initialize the starting and ending index of the current batch
            start_idx = i*batch_size
            end_idx = start_idx+batch_size
            batch_x = rand_x[start_idx:end_idx,:]
            batch_y = rand_y[start_idx:end_idx]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            # print statistics
            curr_loss = loss.item()

        # print statistics every 5 epochs
        if show_loss and (e % 5 == 0):
            print(f"Epoch {e+5}, current loss: {curr_loss}")


def test_model(model, test_x, test_y):
    """
    Test the already trained model using the testing set

    @param: test_x: torch.Tensor, test_y: torch.Tensor
    @return: None
    """
    # get the accuracy of the model on the testing set
    percent_correct = fPC(model, test_x, test_y)

    # print out the accuracy
    print(f'Accuracy of the model on testing set: {percent_correct * 100} %')


def main():
    """
    Main function.
    """
    # load the datasets
    train_x, train_y = load_data('UCI_HAR_Dataset', 'train')
    test_x, test_y = load_data('UCI_HAR_Dataset', 'test')

    # initialize hyperparameters
    hidden_size = 512
    epochs = 200
    batch_size = 150
    learning_rate = 0.005
    momentum = 0.9

    # initialize the model and optimizer with cross entropy loss function
    model = FF(561, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # train the model
    train_model(model, optimizer, criterion, train_x, train_y, epochs, batch_size, show_loss=True)

    # test the model
    test_model(model, test_x, test_y)

    # save the model
    # torch.save(model.state_dict(), './g_model.pth')


if __name__ == "__main__":
    main()

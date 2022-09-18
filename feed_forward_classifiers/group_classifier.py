# Native libraries
import os
# External libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FF(nn.Module):
    """
    Class for feed-forward model from pytorch.
    """

    def __init__(self, input_size, hidden_size):
        """
        Class constructor for feed-forward model.

        @param: input_size: int, hidden_size: int
        @return: None
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 6)


    def forward(self, data):
        """
        Forward propagation for the model.

        First layer converts input data into the specified hidden layer size.
        Second layer takes the output of the first layer through ReLU activation function.
        Third layer converts hidden layer into 6 classes from the dataset

        @param: data: torch.Tensor
        @return: data: torch.Tensor
        """
        data = self.fc1(data)
        data = F.relu(data)
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


def fPC(model, data, labels, class_stats=False):
    """
    Measure of percent correct of the current model

    @param: data: torch.Tensor, labels: torch.Tensor
    @return: percent_correct: float
    """
    # initialize class predictions statistics
    classes = ['WALKING', 'U-STAIRS', 'D-STAIRS', 'SITTING', 'STANDING', 'LAYING']
    correct_predictions = np.array([0, 0, 0, 0, 0, 0])
    total_predictions = np.array([0, 0, 0, 0, 0, 0])

    # run model without gradient calculation for better performance
    with torch.no_grad():
        # run the data through the model
        logits = model(data)
        outputs = F.softmax(logits, dim=1)
        # get the predicted result from the output of the model
        _, predicted = torch.max(outputs.data, 1)
        # get the number of correct guesses
        correct = (predicted == labels).sum().item()
        # calculate the percent correct
        percent_correct = (correct / labels.size(0))

        # if class statistics flag is set, calculate the accuracy for each class
        if class_stats:
            # for each truth-guess pair, increment the correct/total predictions
            for truth, guess in zip(labels, predicted):
                correct_predictions[truth] += 1 if truth == guess else 0
                total_predictions[truth] += 1

            # calculate the class accuracies
            class_acc = (correct_predictions / total_predictions)
            # print out the class accuracies
            for i in range(len(classes)):
                print(f"Class {classes[i]}:\t{class_acc[i]*100}%")

    return percent_correct


def optimize_hyperparameters(validation_x, validation_y, count):
    """
    Optimize hyperparameters on the validation set

    @param: validation_x: torch.Tensor, validation_y: torch.Tensor, count: int
    @return: best_hp: dict
    """
    # initialize the hyperparameters options
    hidden_layer_list = np.array([64, 128, 256, 512, 1024])
    batch_list = np.array([50, 100, 150, 200, 250])
    learn_rate_list = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
    epoch_list = np.array([10, 25, 50, 75, 100])
    momentum_list = np.array([0.1, 0.25, 0.5, 0.75, 1])

    # initialize a dictionary to store the optimal hyperparameters
    best_hp = {
        'hidden': 0.0,
        'batch': 0.0,
        'learn_rate': 0.0,
        'num_epoch': 0.0,
        'momentum': 0.0,
        'loss': float('inf'),
        'accuracy': 0.0
    }

    # optimize the hyperparametes over a set count
    print('Optimizing hyperparameters...')
    for i in range(count):
        # randomly choose a set of hyperparameters
        hidden_size = np.random.choice(hidden_layer_list)
        epoch = np.random.choice(epoch_list)
        batch_size = np.random.choice(batch_list)
        learning_rate = np.random.choice(learn_rate_list)
        momentum = np.random.choice(momentum_list)

        # train the model using given hyperparameters
        print(f"Run number: {i+1}\n" \
        f"    Hidden layer size:\t{hidden_size}\n" \
        f"    Epoch:\t\t{epoch}\n" \
        f"    Batch size:\t\t{batch_size}\n" \
        f"    Learning rate:\t{learning_rate}\n" \
        f"    Momentum rate:\t{momentum}")

        # initialize the model and optimizer with cross entropy loss function
        model = FF(validation_x.size(1), hidden_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        loss, acc = train_model(model, optimizer, criterion, validation_x, validation_y, epoch, batch_size)
        print(f"    Loss:\t\t{loss}\n" \
              f"    Accuracy:\t\t{acc}\n")

        # update the optimal hyperparameters if the loss is lower and the accuracy is higher
        if (loss < best_hp['loss']) and (acc > best_hp['accuracy']):
            best_hp.update(hidden = hidden_size,
                           batch = batch_size,
                           learn_rate = learning_rate,
                           num_epoch = epoch,
                           momentum = momentum,
                           loss = loss,
                           accuracy = acc)

    # print out optimal hyperparameters
    print(f"Optimized hyperparameters:\n{best_hp}\n")
    return best_hp


def train_model(model, optimizer, criterion, train_x, train_y, epoch, batch_size, show_loss=False):
    """
    Train the model with the specified hyperparameters

    @param: model: FF, optimizer: torch.optim, criterion: torch.nn.modules.loss
    @return: None
    """
    # keep track of current loss and accuracy for statistics
    epoch_loss = 0.0
    epoch_acc = 0.0

    # loop through the dataset multiple times
    for e in range(epoch):
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

            # set statistics
            epoch_loss = loss.item()
            epoch_acc = fPC(model, batch_x, batch_y)

        # print statistics every 5 epoch
        if show_loss and (e % 5 == 0):
            print(f"Epoch {e+5}\n" \
                  f"    current loss:\t{epoch_loss}\n"\
                  f"    current accuracy:\t{epoch_acc*100}%\n")

    # get the accuracy of the whole dataset
    epoch_acc = fPC(model, train_x, train_y)
    return epoch_loss, epoch_acc


def test_model(model, test_x, test_y):
    """
    Test the already trained model using the testing set

    @param: test_x: torch.Tensor, test_y: torch.Tensor
    @return: None
    """
    # get the accuracy of the model on the testing set
    percent_correct = fPC(model, test_x, test_y, class_stats=True)

    # print out the accuracy
    print(f"Accuracy of the model on testing set: {percent_correct*100}%")


def main():
    """
    Main function.
    """
    # load the datasets
    data_x, data_y = load_data('../UCI_HAR_Dataset', 'train')
    test_x, test_y = load_data('../UCI_HAR_Dataset', 'test')

    # randomize the samples
    rand_idx = np.random.permutation(data_x.size(0))
    rand_x, rand_y = data_x[rand_idx,:], data_y[rand_idx]

    # separate validation set and training set
    ratio = int(data_x.shape[0]*0.7)
    train_x, train_y = rand_x[:ratio,:], rand_y[:ratio]
    validation_x, validation_y = rand_x[ratio:,:], rand_y[ratio:]

    # optimize hyperparametes
    hyperparameters = optimize_hyperparameters(validation_x, validation_y, 25)

    # initialize the model and optimizer with cross entropy loss function
    model = FF(train_x.size(1), hyperparameters['hidden'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=hyperparameters['learn_rate'], momentum=hyperparameters['momentum'])

    # train the model
    train_model(model, optimizer, criterion, train_x, train_y, hyperparameters['num_epoch'], hyperparameters['batch'], show_loss=True)

    # save the model
    # torch.save(model.state_dict(), './group_model.pth')
    # model = FF(train_x.size(1), hyperparameters['hidden'])
    # model.load_state_dict(torch.load('./group_model.pth'))

    # test the model
    test_model(model, test_x, test_y)


if __name__ == "__main__":
    main()

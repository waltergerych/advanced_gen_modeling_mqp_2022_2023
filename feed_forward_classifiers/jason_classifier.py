import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from typing import *
import os



# def load_x_data(file_path: str) -> torch.Tensor:
#     """Load x test and train data into a list of lists of floats. Returns a tensor"""
#     data = []
#     # Parse the data in each row into a list of lists: [[row1_point1, row1_point2], [row2_point1, row1_point2]]
#     with open(file_path) as f:
#         for row in f:
#             data.append([float(x) for x in [observation for observation in row.replace('  ', ' ').split(' ') if observation]])

#     return torch.tensor(data)

# def load_y_data(file_path: str) -> torch.Tensor:
#     """Load y test and train data into a list of ints, much nicer to parse than x data. Also returns a tensor"""
#     data = []
#     with open(file_path) as f:
#         data = [int(x) for x in f]

#     return torch.tensor(data)

# # Load data into tensors
# X_train = load_x_data("./data/UCI_HAR_Dataset/train/X_train.txt")
# train_labels = load_y_data("./data/UCI_HAR_Dataset/train/y_train.txt")
# X_test = load_x_data("./data/UCI_HAR_Dataset/test/X_test.txt")
# test_labels = load_y_data("./data/UCI_HAR_Dataset/test/y_test.txt")

def load_data(dataset: str, dataset_type: str):
    """Load datasets into tensors"""

    # Much easier method of loding data and labels
    x = np.loadtxt(os.path.join(dataset, dataset_type, f"X_{dataset_type}.txt"))
    y = np.loadtxt(os.path.join(dataset, dataset_type, f"y_{dataset_type}.txt"))

    # Numpy to tensor conversion
    data = torch.from_numpy(x).float()
    labels = torch.from_numpy(y).long()

    # Zero index labels
    labels -= 1

    return data, labels


# Load data and labels
X_train = np.loadtxt("data/UCI_HAR_Dataset/test/X_test.txt", float, "X_float.txt")


# Class for CNN
class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 200)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(200, 6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x



def train_model(model, optimizer, criterion, train_x, train_y, epochs, batch_size):
    for epoch in range(epochs):

        running_loss = 0.0

        rand_idx = np.random.permutation(train_x.size(0))
        rand_x, rand_y = train_x[rand_idx, :], train_y[rand_idx]

        for i in range(0, (train_y.size(0)//batch_size)):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = rand_x[start_idx : end_idx, :]
            batch_y = rand_y[start_idx : end_idx]

            # Same steps as tutorial, zero gradients, then forward + backward steps and optimization
            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()

        print(f"Running loss for epoch {epoch}: {running_loss}")


def get_accuracy(model, inputs, labels):
    """Get model accuracy"""
    total_pred = [0]*6
    correct_pred = [0]*6
    classes = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]

    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
    for label, prediction in zip(labels, predictions):
        correct_pred[label] += 1 if label == prediction else 0
        total_pred[label] += 1

    print("Accuracies: \n")

    for i, data in enumerate(zip(correct_pred, total_pred)):
        corr, total = data
        accuracy = 100 * (float(corr) / total)
        print(f"{str(classes[i])}: {str(accuracy)}%")

    total_acc = 100 * (sum(correct_pred) / sum(total_pred))
    print(f"Overall accuracy: {total_acc}%")
    


def main():
    train_x, train_y = load_data("data/UCI_HAR_Dataset", "train")
    test_x, test_y = load_data("data/UCI_HAR_Dataset", "test")

    # Model parameters
    hidden_size = 512
    epochs = 100
    batch_size = 150
    learning_rate = 0.005
    momentum = 0.9

    model = Net(561, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Train and test
    train_model(model, optimizer, criterion, train_x, train_y, epochs, batch_size)

    get_accuracy(model, test_x, test_y)

    # Save model
    PATH = "./cifar_net.pth"
    torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    main()

# # Create model variables
# hidden_size = 58
# epochs = 5
# learning_rate = 0.001
# momentum = 0.9

# model = Net(561, hidden_size)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# model.train()

# for epoch in range(epochs):
#     running_loss = 0.0
#     optimizer.zero_grad()

#     y_pred = model(X_train)

#     loss = criterion(y_pred, train_labels)
#     loss.backward()

#     optimizer.step()
    
#     # print statistics
#     # running_loss += loss.item()
#     # if i % 2000 == 1999:    # print every 2000 mini-batches
#     #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#     #     running_loss = 0.0

# print("Finished training!")

# # Save model
# PATH = "./cifar_net.pth"
# torch.save(model.state_dict(), PATH)

# output = model(X_test)
# predicted = torch.max(output, 1)

# getAccuracy(model, X_test, test_labels)






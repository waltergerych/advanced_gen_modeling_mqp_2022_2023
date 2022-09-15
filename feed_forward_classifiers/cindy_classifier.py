import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from typing import *
import os

### HYPERPARAMETERS
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 2
learning_rate = 0.001
batch_size = 100
### LOAD DATA

def load_datasets(dataset, datatype):

    x = np.loadtxt(os.path.join(dataset, datatype, f"X_{datatype}.txt"))
    y = np.loadtxt(os.path.join(dataset, datatype, f"y_{datatype}.txt"))

    data = torch.from_numpy(x).float()
    print("Data:", data)
    labels = torch.from_numpy(y).long()
    labels -= 1

    return data, labels


### NN CLASS

class FF(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 6)

    def forward(self, out):
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


def calcAccuracy(model, data, labels):
    correct_pred = np.array([0, 0, 0, 0, 0, 0])
    total_pred = np.array([0, 0, 0, 0, 0, 0])

    classes = ['WALKING', 'U-STAIRS', 'D-STAIRS', 'SITTING', 'STANDING', 'LAYING']

    # run model without gradient calculation for better performance
    with torch.no_grad():
        # run the data through the model

        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        for label, predicted in zip(labels, predicted):
            if label == predicted:
                correct_pred[label] += 1
            total_pred[label] += 1

        for index, dp in enumerate(zip(correct_pred, total_pred)):
            correct, total = dp
            acc = 100 * float(correct) / total

            print("Class " + str(classes[index]) + ":\t" + str(acc))

        total_accuracy = sum(correct_pred) / sum(total_pred)

        print("\nTotal Accuracy:\t" + str(total_accuracy))


def train_model(x_train, y_train, x_test, y_test, epochs, batch_size):

    hidden_size = 512
    iterations = epochs
    learning_rate = .001
    momentum = .8

    # Define the model, loss function (criterion), and optimizer
    model = FF(561, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    current_loss = 0.0
    current_acc = 0.0
    running_loss = 0

    # for every cycle
    for epoch in range(iterations):
        optimizer.zero_grad()

        # FORWARD PASS
        y_pred = model(x_train)

        # BACKWARD PASS
        loss = criterion(y_pred, y_train)
        loss.backward()

        optimizer.step()

    print("Training DONE")

    # PATH = './feed_forward2.pth'
    # torch.save(model.state_dict(), PATH)

    running_loss += loss.item()
    if epoch % 500 == 0:
        print("loss: ", running_loss / 500)
        running_loss = 0.0

    output = model(x_test)
    calcAccuracy(model, x_test, y_test)

    PATH = ".cifar_net.pth"
    torch.save(model.state_dict(), PATH)

    output = model(x_test)
    predicted = torch.max(output, 1)



def main():
    x_train, y_train = load_datasets('UCI HAR Dataset/UCI HAR Dataset', 'train')
    x_test, y_test = load_datasets('UCI HAR Dataset/UCI HAR Dataset', 'test')

    print(x_train, y_train, x_test, y_test)

    train_model(x_train, y_train, x_test, y_test, 500, 100)




if __name__ == "__main__":
    main()


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

## LOAD DATA ##

# load x_training data with 562 features per observation
x_train = []
with open("UCI_HAR_Dataset/train/X_train.txt") as f:
    for row in f:
        x_train.append(row.replace('  ', ' ').split(' '))
float_x_train = []
for row in x_train:
    arr = []
    row.pop(0)
    for item in row:
        arr.append(float(item))
    float_x_train.append(arr)

# load y_training data with activity labels
y_train = []
with open("UCI_HAR_Dataset/train/y_train.txt") as f:
    for row in f:
        y_train.append(row)
int_y_train = []
for item in y_train:
    int_y_train.append(int(item))

# load testing data
x_test = []
with open("UCI_HAR_Dataset/test/X_test.txt") as f:
    for row in f:
        x_test.append(row.replace('  ', ' ').split(' '))
float_x_test = []
for row in x_test:
    arr = []
    row.pop(0)
    for item in row:
        arr.append(float(item))
    float_x_test.append(arr)

y_test = []
with open("UCI_HAR_Dataset/test/y_test.txt") as f:
    for row in f:
        y_test.append(row)
int_y_test = []
for item in y_test:
    int_y_test.append(int(item))

# convert data to a tensor
tensor_x_train = torch.tensor(float_x_train)
train_labels = torch.tensor(int_y_train)
tensor_x_test = torch.tensor(float_x_test)
test_labels = torch.tensor(int_y_test)

train_labels = train_labels.long()
test_labels = test_labels.long()

# Index labels at 0 rather than 1
train_labels = torch.sub(train_labels, 1)
test_labels = torch.sub(test_labels, 1)

## DEFINE NN CLASS ##

# Feed Forward class model
class FF(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Function to evaluate model accuracy
def getAccuracy(model, inputs, labels):
    correct_pred = [0, 0, 0, 0, 0, 0]
    total_pred = [0, 0, 0, 0, 0, 0]
    classes = ['WALKING', 'UPSTAIRS', 'D-STAIRS', 'SITTING', 'STANDING', 'LAYING']

    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct_pred[label] += 1
        total_pred[label] += 1
    
    for i, data in enumerate(zip(correct_pred, total_pred)):
        corr, total = data
        accuracy = 100 * float(corr) / total
        print("Class " + str(classes[i]) + ":\t" + str(accuracy))
    total_accuracy = sum(correct_pred) / sum(total_pred)
    print("\nTotal Accuracy:\t" + str(total_accuracy))

## DEFINE MODEL ##

hidden_size = 58
iterations = 500
learning_rate = .001
momentum = .9

# Define the model, loss function (criterion), and optimizer
model = FF(561, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

## TRAIN MODEL ##
model.train()

running_loss = 0

for epoch in range(iterations):

    # Zero parameter gradients
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(tensor_x_train)

    # Backward pass
    loss = criterion(y_pred, train_labels)
    loss.backward()

    # Optimize
    optimizer.step()

    # print statistics
    # running_loss += loss.item()
    # if epoch % 500 == 0:
    #     print("loss: ", running_loss / 500)
    #     running_loss = 0.0

print("Finished training")

PATH = './feed_forward2.pth'
torch.save(model.state_dict(), PATH)

output = model(tensor_x_test)

predicted = torch.max(output, 1)

getAccuracy(model, tensor_x_test, test_labels)

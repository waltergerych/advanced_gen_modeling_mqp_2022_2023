import torch
import torch.nn as nn
import numpy as np
import random

PATH = "UCI_HAR_Dataset/train/"

## Define generator and discriminator ##
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 561)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

## Function to load data
def load_data():
    x_train = []
    with open(PATH + "X_train.txt") as f:
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
    with open(PATH + "y_train.txt") as f:
        for row in f:
            y_train.append(row)
    int_y_train = []
    for item in y_train:
        int_y_train.append(int(item))

    return float_x_train, int_y_train

# Generate fake data
def get_fake(generator, size):
    z = []
    for data in range(size):
        z.append(np.random.uniform(0, 128))
    z = torch.tensor(z)
    fake = generator(z.unsqueeze(1))
    return fake

# Sample real data
def sample(samples, inputs):
    indices = []
    while len(indices) < samples:
        num = np.random.randint(0, 561)
        if num not in indices:
            indices.append(num)
    data = []
    for num in indices:
        data.append(inputs[num].tolist())
    return torch.tensor(data)

## Define the GAN
iterations = 1000
half_batch_size = 2
learning_rate = .001
momentum = .9
hidden_layers = 58
ratio = 1

generator = Generator(1, hidden_layers)
discrim = Discriminator(561, hidden_layers)
inputs, labels = load_data()
inputs, labels = torch.tensor(inputs), torch.tensor(labels)

# Get a tensor for the batch labels
batch_labels = []
for i in range(half_batch_size):
    batch_labels.append(0)
for i in range(half_batch_size):
    batch_labels.append(1)
batch_labels = torch.tensor(batch_labels)

criterion = nn.CrossEntropyLoss()
g_optimizer = torch.optim.SGD(generator.parameters(), lr=learning_rate, momentum=momentum)
d_optimizer = torch.optim.SGD(discrim.parameters(), lr=learning_rate, momentum=momentum)

## Train ##
turn = 0
gcount, dcount = 0, 0
for i in range(iterations):
    
    # Generate fake data
    fake = get_fake(generator, half_batch_size)

    # Sample real data
    real = sample(half_batch_size, inputs)

    # Combine fake and real data
    batch = torch.cat((fake, real), 0)

    # Feed to discriminator
    outputs = discrim(batch)

    # Loss function alternates at ratio:1 G:D where ratio is defined in GAN parameters
    if turn == ratio:
        turn = 0
        # Backpropagate discriminator
        d_loss = criterion(outputs, batch_labels)
        d_loss.backward()
        d_optimizer.step()
        dcount += 1
    else:
        turn += 1
        # Backpropagate generator
        g_loss = criterion(outputs, batch_labels)
        g_loss.backward()
        g_optimizer.step()
        gcount += 1


print("\nFinished Training")

# Show final generator
fake = get_fake(generator, half_batch_size)
print("Final fake data:")
print(fake)

# Show final discriminator
real = sample(half_batch_size, inputs)
outputs = discrim(torch.cat((fake, real), 0))
print(outputs)

for guess in outputs:
    if guess[0] == 1:
        print("Fake")
    elif guess[1] == 1:
        print("Real")

# Show the total times the generator and discriminator were trained, respectively
print("Total generative count: " + str(gcount))
print("Total discriminator count: " + str(dcount))

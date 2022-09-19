import os
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F

PATH = "UCI_HAR_Dataset/train/"

## Define generator and discriminator ##
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 561)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

## Function to load data
def load_data(dataset: str, dataset_type: str):
    """Load datasets into tensors"""

    x = np.loadtxt(os.path.join(dataset, dataset_type, f"X_{dataset_type}.txt"))
    y = np.loadtxt(os.path.join(dataset, dataset_type, f"y_{dataset_type}.txt"))

    # Numpy to tensor conversion
    data = torch.from_numpy(x).float()
    labels = torch.from_numpy(y).float()

    # Zero index labels
    labels -= 1

    return data, labels

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
        num = np.random.randint(0, inputs.size()[0])
        if num not in indices:
            indices.append(num)
    data = []
    for num in indices:
        data.append(inputs[num].tolist())
    return torch.tensor(data)

def train(generator, discrim, inputs, labels, epochs, half_batch_size, learning_rate, momentum, ratio):
    # Get a tensor for the batch labels
    batch_labels = []
    for i in range(half_batch_size):
        batch_labels.append(0.0)
    for i in range(half_batch_size):
        batch_labels.append(1.0)
    batch_labels = torch.tensor(batch_labels)

    criterion = nn.BCELoss()
    g_optimizer = torch.optim.SGD(generator.parameters(), lr=learning_rate, momentum=momentum)
    d_optimizer = torch.optim.SGD(discrim.parameters(), lr=learning_rate, momentum=momentum)

    ## Train ##
    turn = 0
    for epoch in range(epochs):
        
        # Generate fake data
        fake = get_fake(generator, half_batch_size)

        # Sample real data
        real = sample(half_batch_size, inputs)

        # Combine fake and real data
        batch = torch.cat((fake, real), 0)

        # Feed to discriminator
        outputs = discrim(batch)

        # Loss function alternates at ratio:1 = G:D where ratio is defined in GAN parameters
        if turn == ratio:
            turn = 0
            # Backpropagate discriminator
            d_loss = criterion(outputs.flatten(), batch_labels)
            d_loss.backward()
            d_optimizer.step()
        else:
            turn += 1
            # Backpropagate generator
            g_loss = criterion(outputs.flatten(), batch_labels)
            g_loss.backward()
            g_optimizer.step()


    print("\nFinished Training")


def main():
    ## Define the GAN
    epochs = 100
    half_batch_size = 5
    learning_rate = .005
    momentum = .9
    g_hidden_layers = 58
    d_hidden_layers = 58
    ratio = 5

    generator = Generator(1, g_hidden_layers)
    discrim = Discriminator(561, d_hidden_layers)
    inputs, labels = load_data("UCI_HAR_Dataset", "train")

    train(generator, discrim, inputs, labels, epochs, half_batch_size, learning_rate, momentum, ratio)

    # Show final generator
    fake = get_fake(generator, half_batch_size)
    print("Final fake data:")
    print(fake)

    # Show final discriminator
    real = sample(half_batch_size, inputs)
    outputs = discrim(torch.cat((fake, real), 0))
    print("Final output of discriminator")
    # print(outputs)

    for guess in outputs:
        if guess[0] == 0:
            print("Fake")
        elif guess[0] == 1:
            print("Real")

if __name__ == "__main__":
    main()
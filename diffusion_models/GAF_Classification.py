import matplotlib.pyplot as plt
import numpy as np
from helper_plot import hdr_plot_style
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import glob
from PIL import Image
import os
import pickle
from torch.utils.data import Dataset
from datasets import load_dataset
import pathlib

TRAIN = True


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Batch size during training
batch_size = 512


# Number of channels in the training images. For color images this is 3
nc = 3

# Size of feature maps in classifier
ndf = 4

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


train_path = "diffusion_models/GAF_Classification/train"
test_path = "diffusion_models/GAF_Classification/test"

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(32, interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR),
    transforms.CenterCrop(32),
    # transforms.RandomHorizontalFlip(p=0.5)
    ])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(32, interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR),
    transforms.CenterCrop(32),
    ])

train_loader = DataLoader(torchvision.datasets.ImageFolder(train_path, transform=train_transforms), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(torchvision.datasets.ImageFolder(test_path, transform=test_transforms), batch_size=batch_size, shuffle=True)

# Get class names
root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

# CNN
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Flatten()
        )
        self.lin1 = nn.Linear(512, 256)
        self.relu = nn.LeakyReLU(0.2)
        self.lin2 = nn.Linear(256,1)
        self.sig = nn.Sigmoid()
        

    def forward(self, input):
        y = self.main(input)
        y = self.relu(self.lin1(y))
        y = self.lin2(y)
        y = self.sig(y).squeeze(1)
        return y


# Create the Discriminator
cnn = CNNClassifier().to(device)

if TRAIN:
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    cnn.apply(weights_init)

    # Print the model
    # print(cnn)


    # Initialize BCELoss function
    criterion = nn.MSELoss()

    # Setup Adam optimizers for both G and D
    # optimizerD = optim.AdamW(netD.parameters(), lr=lr)
    optimizerD = optim.Adam(cnn.parameters(), lr=lr, betas=(beta1, 0.999))


    # Training Loop

    # Lists to keep track of progress
    D_losses = []
    iters = 0

    # for i, (data_x, data_y) in enumerate(train_loader, 0):
    #     X = data_x.to(device)
    #     Y = data_y.to(device)
    #     print(X.shape)
    #     print(Y.shape)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data_x, data_y) in enumerate(train_loader, 0):
            
            
            X = data_x.to(device)
            Y = data_y.to(device)

            cnn.zero_grad()
            output = cnn(X)
            loss = criterion(output.float(), Y.float())
            loss.backward()
            optimizerD.step()
            

            # Output training stats
            if i % 400 == 0:
                print(f'Epoch {epoch}/{num_epochs}')
                print(f'MSE for Train: {loss}')
                
                val_preds = []
                val_labels = []
                for _, (val_x, val_y) in enumerate(test_loader, 0):
                    val_X = val_x.to(device)
                    val_Y = val_y.to(device)
                    with torch.no_grad():
                        val_output = cnn(val_X)
                    
                    val_preds.append(val_output)
                    val_labels.append(val_Y)
                    
                val_preds = torch.concat(val_preds)
                val_labels =  torch.concat(val_labels)
                
                val_loss = criterion(val_preds, val_labels)
                print(f'Test MSE: {val_loss}')
                    
                    
                
                print()

            # Save Losses for plotting later
            D_losses.append(loss.item())
    

            iters += 1

    torch.save(cnn.state_dict(), "diffusion_models/classifiers/GAF_Classifier.pth")

else:
    cnn.load_state_dict(torch.load("diffusion_models/classifiers/GAF_Classifier.pth"))

def check_accuracy(test_loader: DataLoader, model: nn.Module, device):
    num_correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device=device)
            labels = labels.to(device=device)
            print(f"Looking at: {labels}")

            predictions = model(data)
            print(f"Model predicted: {predictions}")
            num_correct += (predictions == labels).sum()
            total += labels.size(0)

        print(f"Test Accuracy of the model: {float(num_correct)/float(total)*100:.2f}")

check_accuracy(test_loader, cnn, "cpu")

# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
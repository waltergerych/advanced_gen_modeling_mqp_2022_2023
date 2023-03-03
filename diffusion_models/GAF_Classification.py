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



exit()


class cifar10(Dataset):
	def __init__(self, root, train = False, transforms = None):

		self.root = root
		self.transforms = transforms
		self.train = train


		self.train_data = [file for file in os.listdir(root) if "data_batch" in file] # TODO make test and train
		self.test_data = [file for file in os.listdir(root) if "test_batch" in file]

		self.data_files = self.train_data if self.train else self.test_data
		self.images = []
		self.labels = []
		
		self.load_data()


	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):

		if torch.is_tensor(idx):
			idx = idx.tolist()

		image = self.images[idx]
		image = Image.fromarray(image)

		label = self.labels[idx]

		if self.transforms:
			image = self.transforms(image)

		return image, label


	def load_data(self):

		for file in self.data_files:
			file_path = os.path.join(self.root, file)
			sample = self.read_file(file_path)
			self.images.append(sample["data"])
			self.labels.extend(sample["labels"])


		self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
		self.images = self.images.transpose((0, 2, 3, 1))

	def read_file(self, filename):
		with open(filename, "rb") as f:
			f = pickle.load(f, encoding = "latin1")
		return f





transformations = transforms.Compose([
    # transforms.PILToTensor()
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# image_list = []
# for filename in glob.glob('diffusion_models/GAF_RGB_Images/*.jpg'):
#     im = Image.open(filename)
#     image_list.append(transformations(im))

# print(f"Total images: {len(image_list)}")

# # Create an instance for training. 
# # When we run this code for the first time, the CIFAR10 train dataset will be downloaded locally. 
# # train_set = CIFAR10(root="./data",train=True,transform=transformations,download=True)
# train_set = image_list[:int(0.7*len(image_list))]
# print(f"Length of train set: {len(train_set)}")


# if __name__ == "__main__":
        
#     import torch
#     import torchvision
#     import torchvision.transforms as transforms


#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     batch_size = 4

#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                             download=True, transform=transform)
#     print(trainset)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                             shuffle=True, num_workers=2)

#     testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                         download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                             shuffle=False, num_workers=2)

#     classes = ('plane', 'car', 'bird', 'cat',
#             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



#     import torch.nn as nn
#     import torch.nn.functional as F


#     class Net(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.conv1 = nn.Conv2d(3, 6, 5)
#             self.pool = nn.MaxPool2d(2, 2)
#             self.conv2 = nn.Conv2d(6, 16, 5)
#             self.fc1 = nn.Linear(16 * 5 * 5, 120)
#             self.fc2 = nn.Linear(120, 84)
#             self.fc3 = nn.Linear(84, 10)

#         def forward(self, x):
#             x = self.pool(F.relu(self.conv1(x)))
#             x = self.pool(F.relu(self.conv2(x)))
#             x = torch.flatten(x, 1) # flatten all dimensions except batch
#             x = F.relu(self.fc1(x))
#             x = F.relu(self.fc2(x))
#             x = self.fc3(x)
#             return x


#     net = Net()



#     import torch.optim as optim

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



#     for epoch in range(1):  # loop over the dataset multiple times

#         running_loss = 0.0
#         for i, data in enumerate(trainloader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # print statistics
#             running_loss += loss.item()
#             if i % 2000 == 1999:    # print every 2000 mini-batches
#                 print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#                 running_loss = 0.0

#     print('Finished Training')



#     dataiter = iter(testloader)
#     images, labels = next(dataiter)


#     correct = 0
#     total = 0
#     # since we're not training, we don't need to calculate the gradients for our outputs
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             # calculate outputs by running images through the network
#             outputs = net(images)
#             # the class with the highest energy is what we choose as prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
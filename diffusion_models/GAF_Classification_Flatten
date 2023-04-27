import numpy as np
from sklearn.metrics import precision_score, recall_score
import torch
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
#from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
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
num_epochs = 10

# Learning rate for optimizers
lr = 0.001

# Beta1 hyperparam for Adam optimizers
# why 0.5???
beta1 = 0.5


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# we're working on this rn
# make sure directory is the same on turing
train_path = "GAF_RGB_Class_Images/train"
test_path = "GAF_RGB_Class_Images/test"


def check_accuracy(test_loader: DataLoader, model: nn.Module, device):
    num_correct = 0
    total = 0
    model.eval()

    # Define empty arrays to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device=device) 
            labels = labels.to(device=device)
            # print(f"Looking at: {labels}")

            predictions = model(data)
            predictions = torch.softmax(predictions, dim=1)
            
            # print(f"Model predicted: {predictions}")

            # call argmax to find the guess (which class) and then calc acc
            class_predictions = torch.add(torch.argmax(predictions, dim=1), 1)
            num_correct += (class_predictions == labels).sum()
            total += labels.size(0)

        precision = precision_score(labels, class_predictions, average='macro', zero_division=1)
        recall = recall_score(labels, class_predictions, average='macro', zero_division=1)
        f1score = f1_score(labels, class_predictions, average='macro', zero_division=1)
        
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1score)

        print(f"Test Accuracy of the model: {float(num_correct)/float(total)*100:.2f}%")


train_transforms = transforms.Compose([
    transforms.ToTensor(),
    
    ])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    
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
        self.lin1 = nn.Linear(73728, 256)
        self.relu = nn.LeakyReLU(0.2)
        self.lin2 = nn.Linear(256,6)

    def forward(self, input):

        y = self.main(input)
        y = self.lin1(y)
        y = self.relu(y)
        y = self.lin2(y)

        return y


# Create the Discriminator
cnn = CNNClassifier().to(device)

if TRAIN:
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    cnn.apply(weights_init)

    criterion = nn.CrossEntropyLoss()
    

    # Setup Adam optimizers for both G and D
    # optimizerD = optim.AdamW(netD.parameters(), lr=lr)
    optimizerD = optim.Adam(cnn.parameters(), lr=lr, betas=(beta1, 0.999))


    # Training Loop

    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data_x, data_y) in enumerate(train_loader, 0):
            
            # only need if running on turing
            X = data_x.to(device)
            Y = data_y.to(device)

            optimizerD.zero_grad()
            output = cnn(X)
            # convert output to their respective classes
            
            # pred = torch.add(torch.argmax(output), 1)
            loss = criterion(output, Y)
            loss.backward()
            optimizerD.step()
            

            # Output training stats
            if i % 400 == 0:
                print(f'Epoch {epoch}/{num_epochs}')
                print(f'Train Cross-Entropy Loss: {loss}')
                
                val_preds = []
                val_labels = []
                for _, (val_x, val_y) in enumerate(test_loader, 0):
                    val_X = val_x.to(device)
                    val_Y = val_y.to(device)
                    with torch.no_grad():
                        val_output = cnn(val_X)
                        # val_pred = np.argmax(val_output) + 1
                    
                    val_preds.append(val_output)
                    val_labels.append(val_Y)
                    
                val_preds = torch.concat(val_preds)
                val_labels =  torch.concat(val_labels)
                
                val_loss = criterion(val_preds, val_labels)
                print(f'Test Cross-Entropy Loss: {val_loss}')
                check_accuracy(test_loader, cnn, "cpu")               

            # Save Losses for plotting later
            D_losses.append(loss.item())
    

            iters += 1

    torch.save(cnn.state_dict(), "mqp_env/GAF_Classifier_CrossEntropy.pth")

else:
    cnn.load_state_dict(torch.load("mqp_env/GAF_Classifier_CrossEntropy.pth"))

# Initialize a new instance of the CNNClassifier
cnn_classifier = CNNClassifier()

# Move the model to the device
cnn_classifier.to(device)

# Load the saved model weights if needed
# cnn_classifier.load_state_dict(torch.load(saved_model_path))

# Call the check_accuracy function and pass the test data loader and the device
print("CNN!")
check_accuracy(test_loader, cnn_classifier, device)

check_accuracy(test_loader, cnn, "cpu")

# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

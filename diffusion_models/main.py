# Code from https://github.com/azad-academy/denoising-diffusion-model/blob/main/diffusion_model_demo.ipynb

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

from utils import * 
from model import ConditionalModel
from ema import EMA
from evaluate import *
from classifier import *
from diffusion import *
from gan import *


# Set plot style
hdr_plot_style()

# New dataset for GAFs: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones#
# Contains triaxial information for acceleration and gyroscope data
# Shape: [number_of_datapoints, some_number_of_timesteps]
# This means each row of each of these CSVs will be a timeseries of some number of timesteps.

# load the datasets
train_x, train_y = load_data('dataset/UCI_HAR_Dataset', 'train')
test_x, test_y = load_data('dataset/UCI_HAR_Dataset', 'test')

# GAF data
# Option 1: Choose one axis out of the three for each measurement type
# Option 2: Make a GAF for each axis and stack them on top of each other to form an RGB image
# Going with option 1 first to make sure this works properly
# from glob import glob
# for f in glob("../dataset/UCI_HAR_Dataset_Triaxial/train/Inertial_Signals/*/", recursive=True):
    # print(f)
# print(os.path.exists('../dataset/UCI_HAR_Dataset_Triaxial/train/Intertial_Signals'))
total_acc_x_train = np.loadtxt('dataset/UCI_HAR_Dataset_Triaxial/train/Inertial_Signals/total_acc_x_train.txt')
total_acc_y_train = np.loadtxt('dataset/UCI_HAR_Dataset_Triaxial/train/Inertial_Signals/total_acc_y_train.txt')
total_acc_z_train = np.loadtxt('dataset/UCI_HAR_Dataset_Triaxial/train/Inertial_Signals/total_acc_z_train.txt')
CUSTOM_TYPE = "three_values"
custom_acc = np.loadtxt(f'dataset/UCI_HAR_Dataset_Triaxial/train/Inertial_Signals/custom_acc_{CUSTOM_TYPE}.txt')

HAR_data_classes = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]


# classes = ['WALKING', 'U-STAIRS', 'D-STAIRS', 'SITTING', 'STANDING', 'LAYING']

# Make the class specific directories for GAF images
parent_folder = "difusion_models/GAF_Class_Data"
subfolders = []
for HAR_class in HAR_data_classes:
    subfolders.append(f"diffusion_models/GAF_Class_Data/{HAR_class}")
if not os.path.exists(parent_folder):
    os.makedirs(parent_folder)
for folder in subfolders:
    if not os.path.exists(folder):
        os.makedirs(folder)

labels = train_y
dataset = train_x

# Testing Diffusion Pipeline on Gramian Angular Fields

from pyts.image import GramianAngularField

# Define dataset to generate images on (Must be 2D array)
# Using only x axis of total acceleration to test
# task = total_acc_x_train[0:1000]
task = total_acc_z_train

# Define GAF and fit to data
gadf = GramianAngularField(image_size=48, method='difference')
X_gadf = gadf.fit_transform(task)

len_task = len(task)

# Check that the directory to save the images exists or create it
GAF_DIRECTORY = 'diffusion_models/GAF_RGB_Images'

if not os.path.isdir(GAF_DIRECTORY):
    os.makedirs(GAF_DIRECTORY, exist_ok=True)

labels_int = [int(x) for x in labels.numpy()]

# Iterate over the trajectories to create the images
# for i in range(len_task):
#     print(f'\r{i}/{len_task} - {round(i/len_task*100, 2)}%', end='')
#     plt.axis('off')
#     plt.imshow(X_gadf[i], extent=[0, 1, 0, 1], cmap = 'coolwarm', aspect = 'auto',vmax=abs(X_gadf[i]).max(), vmin=-abs(X_gadf[i]).max())
#     # plt.savefig(f'{GAF_DIRECTORY}/{HAR_data_classes[labels_int[i]]}/{i}.jpg', bbox_inches='tight')
#     plt.savefig(f'{GAF_DIRECTORY}/{i}.jpg', bbox_inches='tight')
#     plt.close("all")





########################
# Classification Model #
########################

# Data Preparation
# Loading and normalizing the data.
# Define transformations for the training and test sets
transformations = transforms.Compose([
    # transforms.PILToTensor()
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# We define the batch size of 10
batch_size = 2 # TODO: Must be GCD of the train-test split?
number_of_labels = 6

image_list = []
for filename in glob.glob('diffusion_models/GAF_RGB_Images/*.jpg'): #assuming gif
    im = Image.open(filename)
    image_list.append(transformations(im))

print(f"Total images: {len(image_list)}")

# Create an instance for training. 
# When we run this code for the first time, the CIFAR10 train dataset will be downloaded locally. 
# train_set = CIFAR10(root="./data",train=True,transform=transformations,download=True)
train_set = image_list[:int(0.7*len(image_list))]
print(f"Length of train set: {len(train_set)}")

# Create a loader for the training set which will read the data within batch size and put into memory.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
print("The number of images in a training set is: ", len(train_loader)*batch_size)

# Create an instance for testing, note that train is set to False.
# When we run this code for the first time, the CIFAR10 test dataset will be downloaded locally. 
# test_set = CIFAR10(root="./data", train=False, transform=transformations, download=True)
test_set = image_list[int(0.7*len(image_list)):]
print(f"Length of test set: {len(test_set)}")

# Create a loader for the test set which will read the data within batch size and put into memory. 
# Note that each shuffle is set to false for the test loader.
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
print("The number of images in a test set is: ", len(test_loader)*batch_size)

print("The number of batches per epoch is: ", len(train_loader))
classes = ("Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying")


# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*10*10, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 24*10*10)
        output = self.fc1(output)

        return output

# Instantiate a neural network model 
model = Network()

# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Function to save the model
def saveModel():
    path = "./diffusion_models/image_classifier.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    # imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))


# Let's build our model
train(5)
print('Finished Training')

# Test which classes performed well
testAccuracy()

# Let's load the model we just created and test the accuracy per label
model = Network()
path = "diffusion_models/image_classifier.pth"
model.load_state_dict(torch.load(path))

# Test with batch of images
testBatch()



# from diffusers import DiffusionPipeline

# Unconditional diffusion model
# pipeline = DiffusionPipeline.from_pretrained("google/ddpm-celebahq-256")

# test_img = pipeline(
#     train_data_dir="classification_images_gadf",
#     dataset_name="classification_images_gadf",
#     resolution=64,
#     output_dor="testing_diffusion_output",
#     train_batch_size=16,
#     num_epochs=100,
#     gradient_accumulation_steps=1,
#     learning_rate=1e-4,
#     lr_warmup_steps=500,
#     mixed_precision="no",
#     ).images[0]

# accelerate launch main.py \
#   --train_data_dir "GAF_Testing_Images" \
#   --dataset_name="GAF_Testing_Images" \
#   --resolution=64 \
#   --output_dir="GAF_Output" \
#   --train_batch_size=16 \
#   --num_epochs=100 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=1e-4 \
#   --lr_warmup_steps=500 \
#   --mixed_precision=no \


#   --push_to_hub




# from diffusers import StableDiffusionImg2ImgPipeline
# import requests
# from PIL import Image
# from io import BytesIO

# # load the pipeline
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16
# )

# # let's download an initial image
# url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

# response = requests.get(url)
# init_image = Image.open(BytesIO(response.content)).convert("RGB")
# init_image = init_image.resize((768, 512))

# prompt = "A fantasy landscape, trending on artstation"

# images = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5).images

# images[0].save("fantasy_landscape.png")



# How to convert GAFs back to time series data?
# https://stackoverflow.com/questions/13203017/converting-images-to-time-series





print("Didn't crash")
exit()

# Define the number of features from the dataset to use. Must be 561 or less
NUM_FEATURES = 40
# Number of time steps
NUM_STEPS = 500
# Number of training steps to do in reverse diffusion (epochs)
NUM_REVERSE_STEPS = 10000
# Number of graphs to plot to show the addition of noise over time (not including X_0)
NUM_DIVS = 10 

# Use feature selection to select most important features
feature_selector = SelectKBest(k=NUM_FEATURES)
importance = feature_selector.fit(dataset, labels)
features = importance.transform(dataset)
dataset = torch.tensor(features)

################
### TRAINING ###
################

# Normal diffusion for entire dataset
diffusion = forward_diffusion(dataset, NUM_STEPS, plot=False)
model = reverse_diffusion(dataset, diffusion, NUM_REVERSE_STEPS, plot=False)
torch.save(model.state_dict(), f'./models/test_model.pth')

# Makes diffusion model for each class for the Classifier
models = []
diffusions = []

original_data, original_labels = dataset, labels

for i in range(len(classes)):
    dataset, labels = get_activity_data(original_data, original_labels, i)

    diffusion = forward_diffusion(dataset, NUM_STEPS, plot=False)
    print("Starting training for class " + str(classes[i]))
    model = reverse_diffusion(dataset, diffusion, NUM_REVERSE_STEPS, plot=False)
    models.append(model)
    diffusions.append(diffusion)

    torch.save(model.state_dict(), f'./models/r{NUM_STEPS}_10K_model_best40_{i}.pth')

##################
### EVALUATION ###
##################

labels = test_y
features = importance.transform(test_x)
dataset = torch.tensor(features)

input_size = dataset.shape[1]
num_to_gen = 150
test_train_ratio = .3

# Gan variables
generator_input_size = 128
hidden_size = 512

# Get real data set to be the same size as generated data set
dataset = dataset[:num_to_gen*len(classes)]
labels = labels[:num_to_gen*len(classes)]

# Get data for each class
diffusion_data = []
diffusion_labels = []
gan_data = []
gan_labels = []

# Get denoising variables
ddpm = get_denoising_variables(NUM_STEPS)

for i in range(len(classes)):
    # Load trained diffusion model
    model = ConditionalModel(NUM_STEPS, dataset.size(1))
    model.load_state_dict(torch.load(f'./models/r{NUM_STEPS}_10K_model_best40_{i}.pth'))

    # Load trained GAN model
    generator = Generator(generator_input_size, hidden_size, dataset.size(1))
    generator.load_state_dict(torch.load(f'./gan/generator/G_{classes[i]}.pth'))

    # Get outputs of both models
    diffusion_output = get_model_output(model, input_size, ddpm, NUM_STEPS, num_to_gen)
    gan_output = generate_data([generator], num_to_gen, generator_input_size)
    
    # CODE TO GRAPH 10 PLOTS OF REMOVING NOISE FOR EACH CLASS 
    # --> MUST CHANGE 'get_model_output' to return x_seq rather than x_seq[-1]
    # true_batch, true_labels = get_activity_data(dataset, labels, i)
    # for j in range(0, NUM_STEPS, 10):
    #     perform_pca(true_batch, output[j], f'T{100-j}')
    # perform_pca(true_batch, output[-1], 'T0')
    # plt.show()

    # Add model outputs and labels to lists
    diffusion_data.append(diffusion_output)
    diffusion_labels.append(torch.mul(torch.ones(num_to_gen), i))
    gan_data.append(gan_output[0])
    gan_labels.append(torch.mul(torch.ones(num_to_gen), i))
    print("Generated data for " + str(classes[i]))

# Concatenate data into single tensor
diffusion_data, diffusion_labels = torch.cat(diffusion_data), torch.cat(diffusion_labels)
gan_data, gan_labels = torch.cat(gan_data), torch.cat(gan_labels)

# Do PCA analysis for fake/real and subclasses
pca_with_classes(dataset, labels, diffusion_data, diffusion_labels, classes, overlay_heatmap=True)

# # Show PCA for each class
for i in range(len(classes)):
    true_batch, true_labels = get_activity_data(dataset, labels, i)
    fake_batch, fake_labels = get_activity_data(diffusion_data, diffusion_labels, i)
    perform_pca(true_batch, fake_batch, f'{classes[i]}')
plt.show()

# Machine evaluation for diffusion and GAN data
print('Testing data from diffusion model')
binary_machine_evaluation(dataset, labels, diffusion_data, diffusion_labels, classes, test_train_ratio, num_steps=NUM_STEPS)
multiclass_machine_evaluation(dataset, labels, diffusion_data, diffusion_labels, test_train_ratio)
separability(dataset, diffusion_data, test_train_ratio)

print('Testing data from gan model')
binary_machine_evaluation(dataset, labels, gan_data, gan_labels, classes, test_train_ratio)
multiclass_machine_evaluation(dataset, labels, gan_data, gan_labels, test_train_ratio)
separability(dataset, gan_data, test_train_ratio)


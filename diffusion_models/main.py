# Code from https://github.com/azad-academy/denoising-diffusion-model/blob/main/diffusion_model_demo.ipynb

import matplotlib.pyplot as plt
import numpy as np
from helper_plot import hdr_plot_style
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest

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
train_x, train_y = load_data('../dataset/UCI_HAR_Dataset', 'train')
test_x, test_y = load_data('../dataset/UCI_HAR_Dataset', 'test')

# GAF data
# Option 1: Choose one axis out of the three for each measurement type
# Option 2: Make a GAF for each axis and stack them on top of each other to form an RGB image
# Going with option 1 first to make sure this works properly
# from glob import glob
# for f in glob("../dataset/UCI_HAR_Dataset_Triaxial/train/Inertial_Signals/*/", recursive=True):
    # print(f)
# print(os.path.exists('../dataset/UCI_HAR_Dataset_Triaxial/train/Intertial_Signals'))
total_acc_x_train = np.loadtxt('../dataset/UCI_HAR_Dataset_Triaxial/train/Inertial_Signals/total_acc_x_train.txt')
body_gyro_x_train = np.loadtxt('../dataset/UCI_HAR_Dataset_Triaxial/train/Inertial_Signals/body_gyro_x_train.txt')
CUSTOM_TYPE = "small_noise"
custom_acc = np.loadtxt(f'../dataset/UCI_HAR_Dataset_Triaxial/train/Inertial_Signals/custom_acc_{CUSTOM_TYPE}.txt')


classes = ['WALKING', 'U-STAIRS', 'D-STAIRS', 'SITTING', 'STANDING', 'LAYING']

labels = train_y
dataset = train_x

# Testing Diffusion Pipeline on Gramian Angular Fields

from pyts.image import GramianAngularField

# Define dataset to generate images on (Must be 2D array)
# Using only x axis of total acceleration to test
# task = total_acc_x_train[0:1000]
task = custom_acc

# Define GAF and fit to data
gadf = GramianAngularField(image_size=48, method='difference')
X_gadf = gadf.fit_transform(task)

len_task = len(task)

# Check that the directory to save the images exists or create it
GAF_DIRECTORY = 'GAF_Testing_Images_Custom'
if not os.path.isdir(GAF_DIRECTORY):
    os.makedirs(GAF_DIRECTORY, exist_ok=True)

# Iterate over the trajectories to create the images
for i in range(len_task):
    print(f'\r{i}/{len_task} - {round(i/len_task*100, 2)}%', end='')
    plt.axis('off')
    plt.imshow(X_gadf[i], extent=[0, 1, 0, 1], cmap = 'coolwarm', aspect = 'auto',vmax=abs(X_gadf[i]).max(), vmin=-abs(X_gadf[i]).max())
    plt.savefig(f'{GAF_DIRECTORY}/{CUSTOM_TYPE}_{i}.jpg', bbox_inches='tight')
    plt.close("all")


from diffusers import DiffusionPipeline

# Unconditional diffusion model
pipeline = DiffusionPipeline.from_pretrained("google/ddpm-celebahq-256")

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


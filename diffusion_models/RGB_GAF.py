from pyts.image import GramianAngularField
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import * 
from model import ConditionalModel
from ema import EMA
from evaluate import *
from classifier import *
from diffusion import *
from gan import *


total_acc_x_train = np.loadtxt('dataset/UCI_HAR_Dataset_Triaxial/test/Inertial_Signals/total_acc_x_test.txt')
total_acc_y_train = np.loadtxt('dataset/UCI_HAR_Dataset_Triaxial/test/Inertial_Signals/total_acc_y_test.txt')
total_acc_z_train = np.loadtxt('dataset/UCI_HAR_Dataset_Triaxial/test/Inertial_Signals/total_acc_z_test.txt')

# combined = []
# for i in range(len(total_acc_x_train)):
#     combined.append([[total_acc_x_train[i]],[total_acc_y_train[i]],[total_acc_z_train[i]]])

# print(combined)

_, labels = load_data('dataset/UCI_HAR_Dataset', 'train')
labels_int = [int(x) for x in labels.numpy()]

HAR_data_classes = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]



# Make the class specific directories for GAF images
parent_folder = "diffusion_models/GAF_RGB_Class_Images_Testing"
subfolders = []
for HAR_class in HAR_data_classes:
    subfolders.append(f"{parent_folder}/{HAR_class}")
if not os.path.exists(parent_folder):
    os.makedirs(parent_folder)
for folder in subfolders:
    if not os.path.exists(folder):
        os.makedirs(folder)



for i in range(len(total_acc_x_train)):
    # R, G, B - X, Y, Z

    print(f'\r{i}/{len(total_acc_x_train)} - {round(i/len(total_acc_x_train)*100, 2)}%', end='')

    # Load the three images
    red_channel = Image.open(f'diffusion_models/Triaxial_GAFs/total_x/{i}.jpg').convert('L')
    green_channel = Image.open(f'diffusion_models/Triaxial_GAFs/total_y/{i}.jpg').convert('L')
    blue_channel = Image.open(f'diffusion_models/Triaxial_GAFs/total_z/{i}.jpg').convert('L')

    # Convert the images to numpy arrays
    array1 = np.array(red_channel)
    array2 = np.array(green_channel)
    array3 = np.array(blue_channel)

    # Stack the arrays along the third dimension to create an RGB image
    rgb_image = np.stack((array1, array2, array3), axis=2)

    # Convert the numpy array to an image and save it
    result_image = Image.fromarray(rgb_image, mode='RGB')
    result_image.save(f'{parent_folder}/{HAR_data_classes[labels_int[i]]}/{i}.png')





# def create_gaf(X):
#     # Normalize the time series data between -1 and 1
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     X = scaler.fit_transform(X)

#     # Calculate the Gramian Angular Summation Field (GASF)
#     XX = np.dot(X, X.T)
#     X_cos = np.sqrt(XX)
#     X_sin = np.sqrt(np.abs(XX - np.diag(np.diag(XX))))
#     X_gram = np.outer(X_cos, X_cos) - np.outer(X_sin, X_sin)

#     # Normalize the GASF between 0 and 1
#     min_ = np.min(X_gram)
#     max_ = np.max(X_gram)
#     X_gram = (X_gram - min_) / (max_ - min_)

#     # Convert the GASF to an RGB image
#     image = np.zeros((X_gram.shape[0], X_gram.shape[1], 3))
#     image[:,:,0] = X_gram
#     image[:,:,1] = X_gram
#     image[:,:,2] = X_gram

#     return image


# df1 = pd.read_csv('data_1.csv', header=None)
# df2 = pd.read_csv('data_2.csv', header=None)
# df3 = pd.read_csv('data_3.csv', header=None)

# Example data
# X = np.random.rand(3, 20) # Example triaxial time series data

# X = combined
# for datapoint in X:

#     X_norm = (datapoint - np.mean(datapoint, axis=0)) / np.std(datapoint, axis=0)

#     # print(X_norm)

#     gasf = GramianAngularField(image_size=24, method='summation')
#     gasf.fit(X_norm)

#     print(X_norm)

#     image = gasf.transform(X_norm)
#     image_rgb = np.stack((image[:,:,0], image[:,:,1], image[:,:,2]), axis=2)

#     example_image = image_rgb[0]

#     # Plot the example image
#     plt.imshow(example_image)
#     plt.show()



# exit()

# for row in range(2):
#     # Reshape the data
#     X = np.vstack([total_acc_x_train[row], total_acc_y_train[row], total_acc_z_train[row]])
#     X = X.reshape(len(X), len(total_acc_x_train), 1)

#     # Create the Gramian Angular Field
#     gasf = GramianAngularField(image_size=5, method='summation')
#     gramian = gasf.fit_transform(X)

#     # Plot the Gramian Angular Field
#     plt.imshow(gramian.squeeze(), cmap='gray')
#     plt.show()
    
    
    
    # X = np.concatenate((total_acc_x_train, total_acc_y_train, total_acc_z_train), axis=1)

    # gaf_image = create_gaf(X)
    # print(gaf_image)
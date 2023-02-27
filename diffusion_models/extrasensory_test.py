import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from utils import *
from model import ConditionalModel
from model import ConditionalTabularModel
from ema import EMA
from evaluate import *
from classifier import *
from diffusion import *
from gan import *

# User ID for ExtraSensory dataset
uid = '1155FF54-63D3-4AB2-9863-8385D0BD0A13'
df, sensor_data, labels = read_user_data(uid)

features = df.columns.tolist()
data = torch.tensor(df.values)

# Variables for diffusion
NUM_STEPS = 100         # Low for testing to speed up
NUM_REVERSE_STEPS = 300
LEARNING_RATE = .0001
BATCH_SIZE = 128
HIDDEN_SIZE = 128
diffusion = get_denoising_variables(NUM_STEPS)

# Separate the continuous and discrete data
continuous, discrete = separate_tabular_data(data, features)

# Select one feature, just for initial testing
discrete = torch.squeeze(discrete[:, 4])        # Prob [.6775, .3225]

# New testing data rather than real data --> Trying to get this working first
test_data = []
w1 = torch.tensor([.95, .05])
w2 = torch.tensor([.2, .8])
num_samples = 1000
test_data.append(torch.multinomial(w1, num_samples, replacement=True))
test_data.append(torch.multinomial(w2, num_samples, replacement=True))
discrete = torch.stack(test_data, dim=1)

test_cont_data = []
test_cont_data.append(torch.multiply(torch.randn(num_samples), torch.randn(num_samples)))
test_cont_data.append(torch.randn(num_samples) * .25 + 3)
continuous = torch.stack(test_cont_data, dim=1)

feature_indices = []
k = 0
for i in range(discrete.shape[1]):
    num = get_classes(discrete[:, i]).shape[0]
    feature_indices.append((k, k + num))
    k += num

# Declare model
model = ConditionalTabularModel(NUM_STEPS, HIDDEN_SIZE, continuous.shape[1], k)
# model.load_state_dict(torch.load(f'./models/tabular_{NUM_STEPS}.pth'))
model, loss, probs = reverse_tabular_diffusion(discrete, continuous, diffusion, k, feature_indices, BATCH_SIZE, LEARNING_RATE, NUM_REVERSE_STEPS, plot=False, model=model)
torch.save(model.state_dict(), f'./models/tabular_{NUM_STEPS}.pth')

continuous_output, discrete_output = get_tabular_model_output(model, k, 128, feature_indices, continuous, diffusion)
print(discrete_output)
print(continuous_output)
separability(continuous, continuous_output, train_test_ratio=.7)

x = range(NUM_REVERSE_STEPS)
plt.plot(x, loss)
plt.show()

probs = torch.stack(probs)

x = range(NUM_REVERSE_STEPS)
plt.plot(x, probs)
plt.legend(['f1/c1','f1/c2','f2/c1','f2/c2'])
plt.show()

true_data_sample = continuous[:128, :]
x1 = continuous_output[:, 0]
y1 = continuous_output[:, 1]
x2 = true_data_sample[:, 0]
y2 = true_data_sample[:, 1]

plt.scatter(x1, y1, c='blue')
plt.scatter(x2, y2, c='green')
plt.legend(['fake', 'real'])
plt.show()

"""
NOTES:
    
"""

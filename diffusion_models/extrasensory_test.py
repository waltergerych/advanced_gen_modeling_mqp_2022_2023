import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from utils import * 
from model import ConditionalModel
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
NUM_STEPS = 1000         # Low for testing to speed up
NUM_REVERSE_STEPS = 1000
LEARNING_RATE = .0001
BATCH_SIZE = 128
HIDDEN_SIZE = 128
diffusion = get_denoising_variables(NUM_STEPS)

# Separate the continuous and discrete data
continuous, discrete = separate_tabular_data(data, features)

# Select one feature, just for initial testing
discrete = torch.squeeze(discrete[:, 4])        # Prob [.6775, .3225]

# New testing data rather than real data --> Trying to get this working first
weights = torch.tensor([.8, .2])
num_samples = 1000
discrete = torch.multinomial(weights, num_samples, replacement=True)

k = get_classes(discrete).shape[0]
k = 2

# Declare model
model = ConditionalMultinomialModel(NUM_STEPS, HIDDEN_SIZE, k)   # Need to declare as number of classes for feature. Multiple features????
# model.load_state_dict(torch.load(f'./models/discrete_{NUM_STEPS}.pth'))
model, loss, probs = reverse_tabular_diffusion(discrete, features, diffusion, BATCH_SIZE, LEARNING_RATE, NUM_REVERSE_STEPS, plot=False, model=model)
torch.save(model.state_dict(), f'./models/discrete_{NUM_STEPS}.pth')

output = get_discrete_model_output(model, k, diffusion, 128)
print(output)

x = range(NUM_REVERSE_STEPS)
plt.plot(x, loss)
plt.show()

probs = torch.stack(probs)

x = range(NUM_REVERSE_STEPS)
plt.plot(x, probs)
plt.show()

"""
NOTES:
    Currently working on just discrete diffusion with ConditionalMultinomialModel.
    
    Next Steps:
        - Get model to learn/work with test data with one feature
        - Get model to learn/work with test data with multiple features (concat feature one-hots)
        - Then get model to learn with real ExtraSensory dataset


    Multinomial Diffusion: https://arxiv.org/pdf/2102.05379.pdf (Section 4)

    Current State: Training loss is not decreasing.  Potentially fixes/issues
        - Any hyperparameters (batch_size, training steps (epochs))
        - Incorrect loss function (pretty sure its right)
        - Incorrect model architecture
"""

"""
Questions for Walter

- Can you build a Linear Layer of 0 size?
Ex: self.lin = nn.Linear(128, cont_size) with cont_size = 0

Reasoning --> Make two models/layers, one for continuous and one for discrete.  
If data happens to have no discrete features, size would be (128, 0)

"""

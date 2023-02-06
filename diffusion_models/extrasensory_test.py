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
NUM_STEPS = 100         # Low for testing to speed up
NUM_REVERSE_STEPS = 200
diffusion = get_denoising_variables(NUM_STEPS)

# Separate the continuous and discrete data
continuous, discrete = separate_tabular_data(data, features)

# Declare model
model = ConditionalMultinomialModel(NUM_STEPS, discrete.size(1))
model, loss = reverse_tabular_diffusion(data, features, diffusion, NUM_REVERSE_STEPS, plot=False, model=model)
torch.save(model.state_dict(), f'./models/discrete_{NUM_STEPS}.pth')
print(loss)

x = range(NUM_REVERSE_STEPS)
plt.plot(x, loss)
plt.show()

"""
NOTES:
    Currently working on just discrete diffusion with ConditionalMultinomialModel.
    Testing using ExtraSensory dataset and extracting only discrete features
    
    Goal: get reverse diffusion to work with discrete data
    Includes: loss function, working model architecture, training loop, 
             and any sort of evaluation for it (plotting discrete distributions?)
             to show that the original data is recovered

    Multinomial Diffusion: https://arxiv.org/pdf/2102.05379.pdf (Section 4)

    Current State: Loss function may not be working? Loss does not seem to go down
            and model does not seem to be learning.  Loss graphed for latest training 
            with only 200 steps saved in /training_loss folder
"""

"""
Questions for Walter

- Can you build a Linear Layer of 0 size?
Ex: self.lin = nn.Linear(128, cont_size) with cont_size = 0

Reasoning --> Make two models/layers, one for continuous and one for discrete.  
If data happens to have no discrete features, size would be (128, 0)

"""

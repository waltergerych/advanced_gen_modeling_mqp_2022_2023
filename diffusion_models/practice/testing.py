import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_checkerboard,make_circles,make_moons,make_s_curve,make_swiss_roll
from helper_plot import hdr_plot_style
import torch
import torch.optim as optim
from utils import * 

from model import ConditionalModel
from ema import EMA
from evaluate import *
import seaborn as sns
import numpy as np

data_ = np.random.randn(8,12)
ax = sns.heatmap(data_)
plt.show()
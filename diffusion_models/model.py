#An Implementation of Diffusion Network Model
#Oringinal source: https://github.com/acids-ircam/diffusion_models

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import separate_tabular_data

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out

# Vanilla diffusion model (continuous data)
class ConditionalModel(nn.Module):
    def __init__(self, n_steps, input_size):
        super(ConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(input_size, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, input_size)
    
    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        return self.lin4(x)

# Tabular diffusion model --> TODO
class ConditionalTabularModel(nn.Module):
    def __init__(self, n_steps, input_size, cont_size, cat_size):
        super(ConditionalTabularModel, self).__init__()
        self.lin1 = ConditionalLinear(input_size, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, cont_size)
        self.lin5 = nn.Linear(128, cat_size)
    
    def forward(self, x_c, x_d, y):
        x = torch.stack([x_c, x_d], dim=-1)
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        return self.lin4(x_c), self.lin5(x_d)

# Multinomial diffusion model
class ConditionalMultinomialModel(nn.Module):
    def __init__(self, n_steps, input_size):
        super(ConditionalMultinomialModel, self).__init__()
        self.lin1 = ConditionalLinear(input_size, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, input_size)
    
    def forward(self, x, y):
        x = x.squeeze(1)
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        x = self.lin4(x)
        return F.softmax(x, dim=1)

"""
With multiple features, code would look something like this
# Pass in list of lists to forward method
# results = []
# for indices:
#     result.append(F.softmax(x[start:end]))
# return torch.concat(results, dim=1)
"""
        

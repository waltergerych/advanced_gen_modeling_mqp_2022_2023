# Loss Function for Diffusion Model
# Original Source: https://github.com/acids-ircam/diffusion_models

# Native libraries
import os
# External libraries
import numpy as np
import torch
import pandas as pd
from random import choices

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas

def extract(input, t, x):
    """Extracts a single value from input at step t and reshapes using x.
    Used in the diffusion process

    Args:
        input (torch.Tensor): the input to extract from
        t (torch.Tensor): a tensor with a single element representing the time step to index the input
        x (torch.Tensor): the real data.  Only used for the shape
    """
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def log_1_min_a(a):
    """Used for calculating categorical noise variables"""
    return torch.log(1 - a.exp() + 1e-40)

def q_posterior_mean_variance(x_0, x_t, t,posterior_mean_coef_1,posterior_mean_coef_2,posterior_log_variance_clipped):
    coef_1 = extract(posterior_mean_coef_1, t, x_0)
    coef_2 = extract(posterior_mean_coef_2, t, x_0)
    mean = coef_1 * x_0 + coef_2 * x_t
    var = extract(posterior_log_variance_clipped, t, x_0)
    return mean, var

def p_mean_variance(model, x, t):
    # Go through model
    out = model(x, t)
    # Extract the mean and variance
    mean, log_var = torch.split(out, 2, dim=-1)
    var = torch.exp(log_var)
    return mean, log_var

def p_sample(model, x, t,alphas,betas,one_minus_alphas_bar_sqrt):
    t = torch.tensor([t])
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_bar_sqrt, t, x))
    # Model output
    eps_theta = model(x, t)
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    # Generate z
    z = torch.randn_like(x)
    # Fixed sigma
    sigma_t = extract(betas, t, x).sqrt()
    sample = mean + sigma_t * z
    return (sample)

def p_sample_loop(model, shape,n_steps,alphas,betas,one_minus_alphas_bar_sqrt):
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i,alphas,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(torch.tensor(np.sqrt(2.0 / np.pi)) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, means, log_scales):
    # Assumes data is integers [0, 255] rescaled to [-1, 1]
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
    log_one_minus_cdf_min = torch.log(torch.clamp(1 - cdf_min, min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(x < -0.999, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(torch.clamp(cdf_delta, min=1e-12))))
    return log_probs

def normal_kl(mean1, logvar1, mean2, logvar2):
    """Calculates KL divergence for loss function"""
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
    return kl

def q_sample(x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt ,noise=None):
    """Samples q(t)"""
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
    return (alphas_t * x_0 + alphas_1_m_t * noise)

def loss_variational(model, x_0,alphas_bar_sqrt, one_minus_alphas_bar_sqrt,posterior_mean_coef_1,posterior_mean_coef_2,posterior_log_variance_clipped,n_steps):
    batch_size = x_0.shape[0]
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,))
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()
    # Perform diffusion for step t
    x_t = q_sample(x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
    # Compute the true mean and variance
    true_mean, true_var = q_posterior_mean_variance(x_0, x_t, t,posterior_mean_coef_1,posterior_mean_coef_2,posterior_log_variance_clipped)
    # Infer the mean and variance with our model
    model_mean, model_var = p_mean_variance(model, x_t, t)
    # Compute the KL loss
    kl = normal_kl(true_mean, true_var, model_mean, model_var)
    kl = torch.mean(kl.view(batch_size, -1), dim=1) / np.log(2.)
    # NLL of the decoder
    decoder_nll = -discretized_gaussian_log_likelihood(x_0, means=model_mean, log_scales=0.5 * model_var)
    decoder_nll = torch.mean(decoder_nll.view(batch_size, -1), dim=1) / np.log(2.)
    # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
    output = torch.where(t == 0, decoder_nll, kl)
    return output.mean(-1)

def noise_estimation_loss(model, x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):
    batch_size = x_0.shape[0]
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,))
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()
    # x0 multiplier
    a = extract(alphas_bar_sqrt, t, x_0)
    # eps multiplier
    am1 = extract(one_minus_alphas_bar_sqrt, t, x_0)
    e = torch.randn_like(x_0)
    # model input
    x = x_0 * a + e * am1
    output = model(x, t)
    return (e - output).square().mean()

def extract_cat(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def q_x_cat(data, t, diffs):
    """Adds noise to categorical data

    Args:
        data (torch.Tensor): the categorical data
        t (torch.Tensor): the number of time steps of noise to add
        diffs (Diffusion): the class encapsulating the diffusion variables
    """
    # Get all categorical classes and the size of the data
    classes = get_classes(data)
    K = classes.shape[0]
    size = data.shape[0]

    # Get probability distribution, add noise, and sample from distribution again
    probs = get_probs(data, classes)
    log_cumprod_alpha_t = extract_cat(diffs.log_cumprod_alpha, t, probs.shape)
    log_1_min_cumprod_alpha = extract_cat(diffs.log_1_min_cumprod_alpha, t, probs.shape)
    log_probs = log_add_exp(probs + log_cumprod_alpha_t, log_1_min_cumprod_alpha - np.log(K))
    data = torch.stack(choices(classes, weights=log_probs, k=size))
    return data

def get_probs(data, K):
    """Calculate probablity distribution for given data with K classes"""
    totals = data.squeeze().bincount(minlength=K)
    return totals / torch.sum(totals)

def get_classes(data):
    """Finds all the classes in the data"""
    return data.unique(return_counts=True)[0]

def get_model_output(model, input_size, diffusion, num_steps, num_to_gen):
    """Gets the output of the model
    
    Args:
        model (ConditionalModel): the model to be used
        num_steps (int): the number of noise steps
        dataset (torch.Tensor): the real data to model after
    """
    with torch.no_grad():
        x_seq = p_sample_loop(model, torch.Size([num_to_gen, input_size]), num_steps, diffusion.alphas, diffusion.betas, diffusion.one_minus_alphas_bar_sqrt)
    output = x_seq[-1]

    return output

def load_data(dataset, dataset_type):
    """Load data from text file

    Load data from a given dataset name and dataset type (train/test).
    The function expects the data to be in the following format:
    "{dataset}/{dataset_type}/(X|y)_{dataset_type}.txt"

    Args:
        dataset (string): the name of the directory that the data lives in.
        dataset_type (string): train or test type

    Returns:
        data (torch.Tensor): the features of the data.
        labels (torch.Tensor): the labels of the data.
    """
    # load data and its labels
    x = np.loadtxt(os.path.join(dataset, dataset_type, f"X_{dataset_type}.txt"))
    y = np.loadtxt(os.path.join(dataset, dataset_type, f"y_{dataset_type}.txt"))

    # convert loaded data from numpy to tensor
    data = torch.from_numpy(x).float()
    labels = torch.from_numpy(y).float()

    # convert 1-indexed class labels to 0-indexed labels
    labels -= 1

    return data, labels


def get_activity_data(x, y, activity_label):
    """Parse through data set to get a specified activity

    Given data x, y, and an activity label, return a subset of the data with only specified label.
    Activity label is defined as the following:
        WALKING: 0
        WALKING_UPSTAIRS: 1
        WALKING_DOWNSTAIRS: 2
        SITTING: 3
        STANDING: 4
        LAYING: 5

    Args:
        x (torch.Tensor): the features of the data.
        y (torch.Tensor): the labels of the data.
        activity_label (int): specify the activity label wanted.

    Returns:
        data_x (torch.Tensor): the features of the data given the activity label.
        data_y (torch.Tensor): the labels of the data given the activity label.
    """
    # find a list of index in y where label is equal to the specified activity label
    activity_idx = (y == activity_label).nonzero().flatten()
    # make data_x and data_y tensor with data from the specified activity label
    data_x = x[activity_idx,:]
    data_y = torch.multiply(torch.ones(data_x.size(0)), activity_label)

    return data_x, data_y

def read_user_data(uid):
    """Reads a user data from the ExtraSensory dataset given a user ID

    Assumes the current folder/file structure does not change
    Example UID: '1155FF54-63D3-4AB2-9863-8385D0BD0A13'

    Args:
        uid (String): the user ID of the user to get the data for

    Returns:
        df (pandas.DataFrame): the dataframe of the user's data
        feature_names (pandas.DataFrame): the data for all of the features
        labels (pandas.DataFrame): the data for all of the labels
    """
    df = pd.read_csv(f'./../../dataset/ExtraSensory/{uid}.features_labels.csv/{uid}.features_labels.csv')
    feature_names = df.iloc[:, 0:226]
    labels = df.iloc[:, 226:]
    return df, feature_names, labels
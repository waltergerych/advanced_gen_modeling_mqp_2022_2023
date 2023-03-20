# Internal libraries
import diffusion as dfn
import evaluate as eval
import utils
from model import ConditionalTabularModel

# External libraries
import matplotlib.pyplot as plt
import torch


def main():
    """Toy dataset for continuous + discrete diffusion model
    """
    # user ID for ExtraSensory dataset
    uid = '1155FF54-63D3-4AB2-9863-8385D0BD0A13'
    df,_,_ = utils.read_user_data(uid)

    features = df.columns.tolist()
    data = torch.tensor(df.values)

    # variables for diffusion
    NUM_STEPS = 1000             # Low for testing to speed up
    NUM_REVERSE_STEPS = 100    # ~Epochs
    LEARNING_RATE = .001
    BATCH_SIZE = 128
    HIDDEN_SIZE = 128
    diffusion = dfn.get_denoising_variables(NUM_STEPS)

    # separate the continuous and discrete data
    continuous, discrete = utils.separate_tabular_data(data, features)

    # select one feature, just for initial testing
    discrete = torch.squeeze(discrete[:, 4])        # Prob [.6775, .3225]

    # new testing data rather than real data --> Trying to get this working first
    test_data = []
    w1 = torch.tensor([.95, .05])
    w2 = torch.tensor([.1, .3, .6])
    num_samples = 1000
    test_data.append(torch.multinomial(w1, num_samples, replacement=True))
    test_data.append(torch.multinomial(w2, num_samples, replacement=True))
    discrete = torch.stack(test_data, dim=1)

    test_cont_data = []
    test_cont_data.append(torch.distributions.Beta(2, 25).sample(sample_shape=torch.Size([num_samples])))
    test_cont_data.append(0 - torch.distributions.Beta(5, 1).sample(sample_shape=torch.Size([num_samples])))
    continuous = torch.stack(test_cont_data, dim=1)
    # plt.scatter(continuous[:, 0], continuous[:, 1])
    # plt.show()

    feature_indices = []
    k = 0
    for i in range(discrete.shape[1]):
        num = utils.get_classes(discrete[:, i]).shape[0]
        feature_indices.append((k, k + num))
        k += num

    # declare model
    model = ConditionalTabularModel(NUM_STEPS, HIDDEN_SIZE, continuous.shape[1], k)
    # model.load_state_dict(torch.load(f'./models/tabular_{NUM_STEPS}.pth'))
    model, loss, probs = dfn.reverse_tabular_diffusion(discrete, continuous, diffusion, k, feature_indices, BATCH_SIZE, LEARNING_RATE, NUM_REVERSE_STEPS, model=model)
    torch.save(model.state_dict(), f'./models/tabular_{NUM_STEPS}.pth')

    continuous_output, discrete_output = utils.get_tabular_model_output(model, k, num_samples, feature_indices, continuous.shape[1], diffusion, calculate_continuous=True)
    print(discrete_output)
    eval.separability(continuous, continuous_output, train_test_ratio=.7)

    x = range(NUM_REVERSE_STEPS)
    plt.plot(x, loss)
    plt.show()

    probs = torch.stack(probs)

    x = range(NUM_REVERSE_STEPS)
    plt.plot(x, probs)
    plt.legend(['f1/c1','f1/c2','f2/c1','f2/c2'])
    plt.show()

    true_data_sample = continuous[:, :]
    x1 = continuous_output[:, 0]
    y1 = continuous_output[:, 1]
    x2 = true_data_sample[:, 0]
    y2 = true_data_sample[:, 1]

    plt.scatter(x1, y1, c='blue')
    plt.scatter(x2, y2, c='green')
    plt.legend(['fake', 'real'])
    plt.show()


if __name__ == "__main__":
    main()

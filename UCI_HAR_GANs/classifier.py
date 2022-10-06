# External libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class Classifier(nn.Module):
    """
    Class for feed-forward model from pytorch.
    """

    def __init__(self, input_size, hidden_size):
        """
        Class constructor for feed-forward model.

        @param: input_size: int, hidden_size: int
        @return: None
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 6)


    def forward(self, data):
        """
        Forward propagation for the model.

        First layer converts input data into the specified hidden layer size.
        Second layer takes the output of the first layer through ReLU activation function.
        Third layer converts hidden layer into 6 classes from the dataset

        @param: data: torch.Tensor
        @return: data: torch.Tensor
        """
        data = self.fc1(data)
        data = F.relu(data)
        data = self.fc2(data)
        return data

def get_accuracy(model, data, labels, class_stats=False):
    """
    Measure of percent correct of the current model

    @param: data: torch.Tensor, labels: torch.Tensor
    @return: percent_correct: float
    """
    # initialize class predictions statistics
    classes = ['WALKING', 'U-STAIRS', 'D-STAIRS', 'SITTING', 'STANDING', 'LAYING']
    correct_predictions = np.array([0, 0, 0, 0, 0, 0])
    total_predictions = np.array([0, 0, 0, 0, 0, 0])
    total_confidence = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    total_guesses = np.array([0, 0, 0, 0, 0, 0])

    # run model without gradient calculation for better performance
    with torch.no_grad():
        # run the data through the model
        logits = model(data)
        outputs = F.softmax(logits, dim=1)
        # get the predicted result from the output of the model
        confidence, predicted = torch.max(outputs.data, 1)
        # get the number of correct guesses
        correct = (predicted == labels).sum().item()
        # calculate the percent correct
        percent_correct = (correct / labels.size(0))

        # if class statistics flag is set, calculate the accuracy for each class
        if class_stats:
            # for each truth-guess pair, increment the correct/total predictions
            for truth, guess, conf in zip(labels, predicted, confidence):
                correct_predictions[truth.item()] += 1 if truth == guess else 0
                total_predictions[truth] += 1
                total_confidence[guess] += conf.item()
                total_guesses[guess] += 1

            # calculate the class accuracies
            class_acc = (correct_predictions / total_predictions)
            # print out the class accuracies
            for i in range(len(classes)):
                print(f"Class {classes[i]}:\t{class_acc[i]*100}%")

            # print out the accuracy
            print(f"Accuracy of the model on testing set: {percent_correct*100}%\n")

            # calculate the average confidence of the guess in each class
            class_conf = (total_confidence / total_guesses)
            # print out the average confidence of each class
            for i in range(len(classes)):
                print(f"Confidence in class {classes[i]} guess:\t{class_conf[i]*100}%")

            # show confusion matrix
            confusion_matrix_df = pd.DataFrame(confusion_matrix(labels, predicted))
            print("\nConfusion Matrix")
            print(confusion_matrix_df)
            sns.heatmap(confusion_matrix_df, annot=True)

            # print classification report
            print(classification_report(labels, predicted))

    return percent_correct

def evaluate(generators, batch_size, input_size, true_data, classifier_path):
    """ Evaluates a generator by testing a trained classifier on true and generated data

    Args:
        generators: a list of generators for each class
        batch_size: the size of the data to create for each class
        input_size: the size of the noise for the generators
        true_data: the true data to test on
        classifier_path: the path to the pytorch classifier model
    """
    classifier = Classifier(len(true_data[0][0]), input_size)
    classifier.load_state_dict(torch.load(classifier_path))

    generated_data_x = []
    generated_data_y = []
    for i, generator in enumerate(generators):
        noise = torch.randn(size=(batch_size, input_size)).float()
        generated_data_x.append(generator(noise))
        generated_data_y.append(torch.mul(torch.ones(batch_size), i))

    generated_x, generated_y = torch.cat(generated_data_x), torch.cat(generated_data_y)

    true_x, true_y = true_data

    print("\n---Classifier Performance On Real Data---\n")
    true_correct = get_accuracy(classifier, true_x, true_y.type(torch.int16), class_stats=True)
    print("\n---Classifier Performance On Fake Data---\n")
    generated_correct = get_accuracy(classifier, generated_x, generated_y.type(torch.int16), class_stats=True)

    return true_correct, generated_correct

def train_model(model, optimizer, criterion, train_x, train_y, epoch, batch_size, show_loss=False):
    """
    Train the model with the specified hyperparameters

    @param: model: FF, optimizer: torch.optim, criterion: torch.nn.modules.loss
    @return: None
    """
    # keep track of current loss and accuracy for statistics
    epoch_loss = 0.0
    epoch_acc = 0.0

    # loop through the dataset multiple times
    for e in range(epoch):
        # randomize the samples
        rand_idx = np.random.permutation(train_x.size(0))
        rand_x, rand_y = train_x[rand_idx,:], train_y[rand_idx]

        # process the epoch batch by batch
        for i in range(0, (train_y.size(0)//batch_size)):
            # initialize the starting and ending index of the current batch
            start_idx = i*batch_size
            end_idx = start_idx+batch_size
            batch_x = rand_x[start_idx:end_idx,:]
            batch_y = rand_y[start_idx:end_idx].type(torch.int64)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward(retain_graph=True)
            optimizer.step()

            # set statistics
            epoch_loss = loss.item()
            epoch_acc = get_accuracy(model, batch_x, batch_y)

        # print statistics every 5 epoch
        if show_loss and (e % 5 == 0):
            print(f"Epoch {e+5}\n")
    return epoch_loss, epoch_acc

def optimize_hyperparameters(validation_x, validation_y, count):
    """
    Optimize hyperparameters on the validation set

    @param: validation_x: torch.Tensor, validation_y: torch.Tensor, count: int
    @return: best_hp: dict
    """
    # initialize the hyperparameters options
    hidden_layer_list = np.array([32, 64, 128, 256, 512])
    batch_list = np.array([100, 200, 500, 1000])
    learn_rate_list = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
    epoch_list = np.array([10, 25, 50, 75, 100])
    momentum_list = np.array([0.1, 0.25, 0.5, 0.75, 1])

    # initialize a dictionary to store the optimal hyperparameters
    best_hp = {
        'hidden': 0.0,
        'batch': 0.0,
        'learn_rate': 0.0,
        'num_epoch': 0.0,
        'momentum': 0.0,
        'loss': float('inf'),
        'accuracy': 0.0
    }

    # optimize the hyperparametes over a set count
    print('Optimizing hyperparameters...')
    for i in range(count):
        # randomly choose a set of hyperparameters
        hidden_size = np.random.choice(hidden_layer_list)
        epoch = np.random.choice(epoch_list)
        batch_size = np.random.choice(batch_list)
        learning_rate = np.random.choice(learn_rate_list)
        momentum = np.random.choice(momentum_list)

        # train the model using given hyperparameters
        print(f"Run number: {i+1}\n" \
        f"    Hidden layer size:\t{hidden_size}\n" \
        f"    Epoch:\t\t{epoch}\n" \
        f"    Batch size:\t\t{batch_size}\n" \
        f"    Learning rate:\t{learning_rate}\n" \
        f"    Momentum rate:\t{momentum}")

        # initialize the model and optimizer with cross entropy loss function
        model = Classifier(validation_x.size(1), hidden_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        loss, acc = train_model(model, optimizer, criterion, validation_x, validation_y, epoch, batch_size)
        print(f"    Loss:\t\t{loss}\n" \
              f"    Accuracy:\t\t{acc}\n")

        # update the optimal hyperparameters if the loss is lower and the accuracy is higher
        if (loss < best_hp['loss']) and (acc > best_hp['accuracy']):
            best_hp.update(hidden = hidden_size,
                           batch = batch_size,
                           learn_rate = learning_rate,
                           num_epoch = epoch,
                           momentum = momentum,
                           loss = loss,
                           accuracy = acc)

    # print out optimal hyperparameters
    print(f"Optimized hyperparameters:\n{best_hp}\n")
    return best_hp

def train_classifier(generators, batch_size, input_size):
    """
    Trains a new classifier using trained generators

    Args:
    generators: a list of generators for each class
    batch_size: the size of the data to create for each class
    input_size: the size of the input noise for the generators
    """
    data_x = []
    data_y = []
    for i, generator in enumerate(generators):
        noise = torch.randn(size=(batch_size, input_size)).float()
        data_x.append(generator(noise))
        data_y.append(torch.mul(torch.ones(batch_size), i))

    combined_x, combined_y = torch.cat(data_x), torch.cat(data_y)

    # hp = optimize_hyperparameters(combined_x, combined_y, 25)
    # Results from one run
    hp = {
        'hidden': 512,
        'batch': 100,
        'learn_rate': .1,
        'num_epoch': 25,
        'momentum': .75,
    }
    classifier = Classifier(combined_x.size(1), 128)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=hp['learn_rate'], momentum=hp['momentum'])

    train_model(classifier, optimizer, criterion, combined_x, combined_y, hp['num_epoch'], hp['batch'], show_loss=False)

    return classifier

# External libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FF(nn.Module):
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

    # run model without gradient calculation for better performance
    with torch.no_grad():
        # run the data through the model
        logits = model(data)
        outputs = F.softmax(logits, dim=1)
        # get the predicted result from the output of the model
        _, predicted = torch.max(outputs.data, 1)
        # get the number of correct guesses
        correct = (predicted == labels).sum().item()
        # calculate the percent correct
        percent_correct = (correct / labels.size(0))

        # if class statistics flag is set, calculate the accuracy for each class
        if class_stats:
            # for each truth-guess pair, increment the correct/total predictions
            for truth, guess in zip(labels, predicted):
                correct_predictions[truth.item()] += 1 if truth == guess else 0
                total_predictions[truth] += 1

            # calculate the class accuracies
            class_acc = (correct_predictions / total_predictions)
            # print out the class accuracies
            for i in range(len(classes)):
                print(f"Class {classes[i]}:\t{class_acc[i]*100}%")
    
    # print out the accuracy
    print(f"Accuracy of the model on testing set: {percent_correct*100}%")

    return percent_correct

def evaluate(true_data, generated_data, classifier_path):
    """ Evaluates a generator by testing a trained classifier on true and generated data

    Args:
        generator: the trained generator model
        true_data: the batch of real data
        generated_data: the batch of fake data
        classifier_path: the path to the pytorch classifier model
    """
    classifier = FF(561, 128)
    classifier.load_state_dict(torch.load(classifier_path))

    true_x, true_y = true_data
    generated_x, generated_y = generated_data

    print("\n---Classifier Performance On Real Data---\n")
    true_correct = get_accuracy(classifier, true_x, true_y.type(torch.int16), class_stats=True)
    print("\n---Classifier Performance On Fake Data---\n")
    generated_correct = get_accuracy(classifier, generated_x, generated_y.type(torch.int16), class_stats=True)

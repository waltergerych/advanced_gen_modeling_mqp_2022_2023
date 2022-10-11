from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

def generate_data_and_labels(fake_data, real_data):
    
    # convert to numpy arrays
    fake_data = fake_data.detach().numpy()
    real_data = real_data.detach().numpy()
    
    # generate fake data labels
    fake_data_labels = np.zeros(fake_data.shape[0])
    # generate real data labels
    real_data_labels = np.ones(real_data.shape[0])

    # concatenate data
    all_data = np.concatenate((fake_data, real_data), axis=0)
    all_labels = np.concatenate((fake_data_labels, real_data_labels), axis=0)

    return all_data, all_labels

def train_regression_model(model, X_train, y_train, X_test, y_test):
    print(f"y_train: {y_train}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tn, fp, fn, tp
    accuracy = accuracy_score(y_test, y_pred)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    return accuracy, precision, recall
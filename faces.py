# Toy problem: detecting smiling face using the GENKI dataset
# preprocessed data courtesy of Allen Downey at Olin College

import sklearn
from sklearn.datasets import *
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


def load_smiles():
    smiles = loadmat('smile_dataset.mat')
    return_val = sklearn.datasets.base.Bunch()
    return_val.data = smiles['X']
    return_val.target = smiles['expressions']
    return return_val


def plot_face(pixel_array):
    M = 24  # 24x24 is the resolution of the photo
    im = np.reshape(pixel_array, (M, M))
    plt.imshow(im.T, cmap='gray')
    plt.xticks([])
    plt.yticks([])


def plot_random_faces(faces):
    """ plot some random faces with their training labels"""
    selects = np.random.random_integers(0, 20000, 16)
    plt.figure()
    for k in range(16):
        plt.subplot(4, 4, k+1)
        plot_face(faces.data[selects[k]])
        if faces.target[k] == 1:
            plt.title('smile')
        else:
            plt.title('ugly')


def illustrate_prediction(model, test_data, test_target):
    """ illustrate the prediction accuracy for the smile detector"""
    selects = np.random.random_integers(0, len(test_data), 16)
    labels = test_target[selects]
    predicts = model.predict(test_data[selects])
    plt.figure()
    for k in range(16):
        plt.subplot(4, 4, k+1)
        plot_face(test_data[selects[k]])
        if predicts[k] == 1:
            plt.title('smile')
        else:
            plt.title('ugly')

        if predicts[k] != labels[k]:
            plt.plot([0, 24], [0, 24], 'r', linewidth=2)
            plt.plot([0, 24], [24, 0], 'r', linewidth=2)

# Simple implementation of logistic regression:
faces = load_smiles()
# note that bad labeling may also happen to training data
plot_random_faces(faces)

# split into training and test data 75% for training and 25 % for test
model = LogisticRegression()  # by default: L2 regularization
split_point = int(np.round(0.75 * len(faces.data)))
training_data = faces.data[:split_point]
training_target = faces.target[:split_point]
test_data = faces.data[split_point:]
test_target = faces.target[split_point:]

# fit logistic model with training set
model.fit(training_data, np.reshape(training_target, (len(training_target),)))

# illustrate on test data
illustrate_prediction(model, test_data, test_target)

# scoring model:
model.score(test_data, test_target)

# The effect of training size on model score:
scores = list()
training_sizes = list()
for training_size in xrange(10, 2000, 20):
    print float(training_size)/N
    model.fit(faces.data[:training_size],
              np.reshape(faces.target[:training_size],
              (training_size,)))
    scores.append(model.score(faces.data[-1000:], faces.target[-1000:]))
    training_sizes.append(training_size)

plt.figure()
plt.plot(training_sizes, scores)
plt.xlabel('Training size')
plt.ylabel('Logistic regression model score')
plt.title('Prediction accuracy as a function of training size')


# model coefficients (fun part):
plt.figure()
plt.imshow(numpy.reshape(model.coef_, (24, 24)).T, cmap='gray')
plt.show()

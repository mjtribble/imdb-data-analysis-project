"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""

# This class implements a k-nearest neighbor using scikit-learn
from sklearn import model_selection
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt


class KNN:
    # This creates a new instance of NN
    # @param data = dictionary of Genre's with associated name codes of actors and directors associated with the genre
    # {genre: [names, ....]}
    def __init__(self, data):
        self.data = data
        print('\n\nQ3: Can we predict a genre based on the actor and director of a film?')
        print('Created K-Nearest Neighbor')
        self.run_knn()

    # This splits up the data into test and training sets
    # Runs sets through a neural net to classify the data.
    def run_knn(self):
        x = []
        y = []

        [[x.append(self.data['Action'][i:i + 50]), y.append(0)] for i in range(0, len(self.data['Action']), 50) if
         len(self.data['Action'][i:i + 50]) == 50]
        [[x.append(self.data['Comedy'][i:i + 50]), y.append(1)] for i in range(0, len(self.data['Comedy']), 50) if
         len(self.data['Comedy'][i:i + 50]) == 50]
        [[x.append(self.data['Drama'][i:i + 50]), y.append(2)] for i in range(0, len(self.data['Drama']), 50) if
         len(self.data['Drama'][i:i + 50]) == 50]

        x = np.array(x)
        y = np.array(y)

        # from sklearn.model_selection import KFold
        # kf = KFold(n_splits=4)
        # kf.get_n_splits(x)
        # print(kf)
        # for train_index, test_index in kf.split(x):
        #     print("TRAIN:", train_index, "TEST:", test_index)
        #
        # x_train, x_test = x[train_index], x[test_index]
        # y_train, y_test = y[train_index], y[test_index]

        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, y, test_size=0.33, random_state=42)

        key_dict = {0: 'Action',
                    1: 'Comedy',
                    2: 'Drama'
                    }

        # print(x_train.shape)
        # print(x_test.shape)
        # print(y_train.shape)
        # print(y_test.shape)

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(x_train, y_train)
        pred = knn.predict(x_test)
        print("Actual: ", [key_dict[i] for i in y_test])
        print("Predicted: ", [key_dict[i] for i in pred])
        print('Accuracy: %lf' % (accuracy_score(y_test, pred) * 100), "%")


class NaiveBase:
    def __init__(self):
        print('Created NaiveBase')

"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""


# This class implements a k-nearest neighbor using scikit-learn
from sklearn import model_selection
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt


class KNN:

    # This creates a new instance of NN
    # @param data = dictionary of Genre's with associated name codes of actors and directors associated with the genre
    # {genre: [names, ....]}
    def __init__(self, data):
        self.data = data
        print('Can we predict a genre based on the actor and director of a film?')
        print('Created K-Nearest Neighbor')
        self.run_knn()

    # This splits up the data into test and training sets
    # Runs sets through a neural net to classify the data.
    def run_knn(self):
        x = []
        y = []

        # x.append(self.data['Action'][:100])
        [[x.append(self.data['Action'][i:i + 5]), y.append('Action')] for i in range(0, len(self.data['Action']), 5)]
        [[x.append(self.data['Comedy'][i:i + 5]), y.append('Comedy')] for i in range(0, len(self.data['Comedy']), 5)]
        [[x.append(self.data['Drama'][i:i + 5]), y.append('Drama')] for i in range(0, len(self.data['Drama']), 5)]
        [[x.append(self.data['Romance'][i:i + 5]), y.append('Romance')] for i in range(0, len(self.data['Romance']), 5)]
        [[x.append(self.data['Sci-Fi'][i:i + 5]), y.append('Sci-Fi')] for i in range(0, len(self.data['Sci-Fi']), 5)]
        [[x.append(self.data['Thriller'][i:i + 5]), y.append('Thriller')] for i in range(0, len(self.data['Thriller']), 5)]

        for k in x:
            print(len(k))
        # print(x, y)
        # x.append(self.data['Comedy'][:100])
        # x.append(self.data['Crime'][:100])
        # x.append(self.data['Drama'][:100])
        # x.append(self.data['Romance'][:100])
        # x.append(self.data['Sci-Fi'][:100])
        # x.append(self.data['Thriller'][:100])
        #
        # y.append('Action')
        # y.append('Comedy')
        # y.append('Crime')
        # y.append('Drama')
        # y.append('Romance')
        # y.append('Sci-Fi')
        # y.append('Thriller')

        # x.append(self.data['Action'][:200])
        # x.append(self.data['Comedy'][:200])
        # x.append(self.data['Crime'][:200])
        # x.append(self.data['Drama'][:200])
        #
        # y.append('Action')
        # y.append('Comedy')
        # y.append('Crime')
        # y.append('Drama')

        # for i in self.data:
        #     print(len(self.data[i]))
            # y.append(key)
        #     x.append(self.data[key])

        # print(y)
        # for i in x:
        #     print(i)
        # x = np.array(x)
        # y = np.array(y)

        # from sklearn.model_selection import KFold
        # kf = KFold(n_splits=4)
        # kf.get_n_splits(x)
        # print(kf)
        # for train_index, test_index in kf.split(x):
        #     print("TRAIN:", train_index, "TEST:", test_index)
        #
        # x_train, x_test = x[train_index], x[test_index]
        # y_train, y_test = y[train_index], y[test_index]
        # # indices = np.random.permutation(len(x))
        # x_train = x[indices[:-10]]
        # y_train = y[indices[:-10]]
        # x_test = x[indices[-10:]]
        # y_test = y[indices[-10:]]
        
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, y, test_size=0.33, random_state=5)

        # print(x_train.shape)
        # print(x_test.shape)
        # print(y_train.shape)
        # print(y_test.shape)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train, y_train)
        predictions = knn.predict(x_test)
        print("Actual: ", y_test)

        # x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        # y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
        #                      np.arange(y_min, y_max, 0.1))
        #
        # f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))
        #
        # for idx, clf, tt in zip(np.product([0, 1], [0, 1]),
        #                         [knn],
        #                         ['KNN (k=7)']):
        #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        #     Z = Z.reshape(xx.shape)
        #
        #     axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
        #     axarr[idx[0], idx[1]].scatter(x[:, 0], x[:, 1], c=y,
        #                                   s=20, edgecolor='k')
        #     axarr[idx[0], idx[1]].set_title(tt)
        #
        # plt.show()

        print("Predicted: ", predictions)


class NaiveBase:
    def __init__(self):

        print('Created NaiveBase')


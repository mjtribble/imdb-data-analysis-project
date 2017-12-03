"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""

# This class implements a k-nearest neighbor using scikit-learn
import numpy as np
import itertools
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

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

        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, y, test_size=0.33, random_state=42)

        key_dict = {0: 'Action',
                    1: 'Comedy',
                    2: 'Drama'
                    }

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(x_train, y_train)
        pred = knn.predict(x_test)

        actual = []
        predicted = []
        accuracy = accuracy_score(y_test, pred) * 100
        mislabled_points = (y_test != pred).sum()
        c_matrix = confusion_matrix(y_test, pred)

        for i in y_test:
            actual.append(key_dict[i])
        for i in pred:
            predicted.append(key_dict[i])

        print("Actual: ", actual)
        print("Predicted: ", predicted)
        print('Accuracy: %lf' % accuracy, "%")
        print("Number of mislabeled points out of a total %d points : %d" % (x.shape[0], mislabled_points))

        # Compute confusion matrix
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(c_matrix,
                              classes=key_dict.values(),
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(c_matrix,
                              classes=key_dict.values(),
                              normalize=True,
                              title='Normalized confusion matrix')

        plt.show()


class NaiveBase:
    def __init__(self, data):
        print("\n\nQ5: Can we predict a director based on the keywords from their movies?")
        print('Using Naive Base Classifier')
        self.data = data
        self.run_nb()

    def run_nb(self):

        x = []
        y = []

        # associate each director with a unique integer
        director_dict = {
            'Steven Spielberg': 0,
            'Martin Scorsese': 1,
            # 'Baz Luhrmann': 2,
            # 'Darren Aronofsky': 3,
            # 'George Lucas': 4,
            # 'Spike Jonze': 5,
            # 'David Fincher': 6,
            # 'Guillermo del Toro': 7,
            # 'Quentin Tarantino': 9,
            # 'Robert Rodriguez': 10,
            'Spike Lee': 11,
            # 'Gus Van Sant': 12,
            'Woody Allen': 13,
            # 'Alfred Hitchcock': 14,
            # 'Sofia Coppola': 15,
            # 'John Hughes': 16,
            # 'Tyler Perry': 17,
            # 'John Singleton': 18,
            # 'Tim Burton': 19
        }

        director_reverse_dict = {
            0: 'Steven Spielberg',
            1: 'Martin Scorsese',
            # 2: 'Baz Luhrmann',
            # 3: 'Darren Aronofsky',
            # 4: 'George Lucas',
            # 5: 'Spike Jonze',
            # 6: 'David Fincher',
            # 7: 'Guillermo del Toro',
            # 9: 'Quentin Tarantino',
            # 10: 'Robert Rodriguez',
            11: 'Spike Lee',
            # 12: 'Gus Van Sant',
            13: 'Woody Allen',
            # 14: 'Alfred Hitchcock',
            # 15: 'Sofia Coppola',
            # 16: 'John Hughes',
            # 17: 'Tyler Perry',
            # 18: 'John Singleton',
            # 19: 'Tim Burton'
        }

        # create a dictionary of every keyword present in the data and set value to 0
        # to be used for one-hot encoding later
        keyword_dict = {}
        for key in self.data:
            for j in self.data[key]:
                if j in keyword_dict:
                    continue
                else:
                    keyword_dict.update({j: 0})

        # split up keywords for each director into lists of 5 keywords.
        # encode a director's name to an unique integer
        # x = list of keyword lists
        # y = list of the int associated with a director via director key
        for key in self.data:
            # print(key, len(self.data[key]))
            director_int = director_dict[key]
            [[x.append(self.data[key][i:i + 15]), y.append(director_int)] for i in range(0, len(self.data[key]), 15) if len(self.data[key][i:i + 15]) == 15]

        # create one-hot encoding for each list of keywords
        x_encoded = []
        for i in x:
            temp_dict = keyword_dict.copy()
            for j in i:
                temp_dict[j] += 1
            x_encoded.append(list(temp_dict.values()))

        # convert from a list to np array
        x_encoded = np.array(x_encoded)
        y = np.array(y)

        # split up training and test sets
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x_encoded, y, test_size=0.33, random_state=42)

        # run gaussian naive bayes on data
        mnb = MultinomialNB()
        mnb.fit(x_train, y_train)
        y_pred = mnb.predict(x_test)

        # determine accuracy and plot
        accuracy = accuracy_score(y_test, y_pred) * 100
        mislabled_points = (y_test != y_pred).sum()
        c_matrix = confusion_matrix(y_test, y_pred)

        actual = []
        predicted = []
        for i in y_pred:
            predicted.append(director_reverse_dict[i])

        for i in y_test:
            actual.append(director_reverse_dict[i])

        print("Actual: ", actual)
        print("Predicted: ", predicted)
        print("\nNumber of mislabeled points out of a total %d points : %d" % (x_encoded.shape[0], mislabled_points))
        print('Accuracy: %lf' % accuracy, "%")


        # Compute confusion matrix
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(c_matrix,
                              classes=director_dict.keys(),
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(c_matrix,
                              normalize=True,
                              classes=director_dict.keys(),
                              title='Normalized confusion matrix')

        plt.show()


# this method creates a confusion matrix and was imported from:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




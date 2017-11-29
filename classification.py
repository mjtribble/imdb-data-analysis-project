"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""


# This class implements a neural network using scikit-learn
class NN:

    # This creates a new instance of NN
    # @param data = dictionary of Genre's with associated name codes of actors and directors associated with the genre
    # {genre: [names, ....]}
    def __init__(self, data):
        self.data = data
        print('Can we predict a genre based on the actor and director of a film?')
        print('Created Neural Net')
        self.run_nn()

    # This splits up the data into test and training sets
    # Runs sets through a neural net to classify the data.
    def run_nn(self):
        x = []
        y = []

        for key in self.data:
            x.append(key)
            y.append(self.data[key])
        print("x", x)
        print("y", y)


class NaiveBase:
    def __init__(self):
        print('Created NaiveBase')


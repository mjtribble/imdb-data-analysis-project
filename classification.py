"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""


# This class implements a k-nearest neighbor using scikit-learn
class KNN:

    # This creates a new instance of NN
    # @param data = dictionary of Genre's with associated name codes of actors and directors associated with the genre
    # {genre: [names, ....]}
    def __init__(self, data):
        self.data = data
        print('Can we predict a genre based on the actor and director of a film?')
        print('Created Neural Net')
        self.run_knn()

    # This splits up the data into test and training sets
    # Runs sets through a neural net to classify the data.
    def run_knn(self):
        data_sets = []
        x = []
        y = []


        for key in self.data:
            # data_sets.append(self.data[key])
            y.append(key)
            x.append(self.data[key])

        print(y)
        print(x)

class NaiveBase:
    def __init__(self):
        print('Created NaiveBase')


"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""

import correlation
import classification
import regression
import pandas as pd
import seaborn as sns

if __name__ == '__main__':

    print('Importing data')
    data = pd.read_csv("INSERT FILE PATH HERE")

    data.head()


def visualize(d, x_vars, y_vars):

    sns.pairplot(d, x_vars, y_vars, size=7, aspect=0.7, kind='reg')

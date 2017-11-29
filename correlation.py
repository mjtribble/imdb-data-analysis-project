"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""

import collections
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr


# This class calculates the Pearson's correlation coefficient and p-value for a given dataset
class Pearson:

    # This creates a new instance of the Pearson Class
    # @param data = dictionary containing two sets of data to analyse
    def __init__(self, data):
        self.data = data  # This is a dictionary {Age: number of roles}
        print('Is there a positive or a negative correlation between age and number of roles an '
              'actor stars in?')
        print("Using Pearson's Correlation Coefficient")
        self.run_pearson()

    # This splits the data into two lists and runs personr on it.
    def run_pearson(self):

        age_list = []
        num_roles_list = []

        # sort the dictionary by age
        od = collections.OrderedDict(sorted(self.data.items()))

        # split dictionary into two lists
        for key in od:
            age_list.append(key)
            num_roles_list.append(od[key])

        # print("age_list: ", age_list)
        # print("num_roles_list: ", num_roles_list)

        # set pcc and pvalue as the values returned from pearsonr()
        pcc, pvalue = pearsonr(age_list, num_roles_list)

        print("Pearson correlation coefficient: ", pcc)
        print("P-value: ", pvalue)


class SpearmanRank:
    def __init__(self):
        print('Created SpearmanRank')



"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""

import collections
from scipy.stats.stats import pearsonr

class Pearson:
    def __init__(self, data):
        self.data = data
        print('Created Pearson: Is there a positive or a negative correlation between age and number of roles an '
              'actor stars in ?')
        self.run_pearson()

    def run_pearson(self):
        age_list = []
        num_roles_list = []
        od = collections.OrderedDict(sorted(self.data.items()))
        for key in od:
            age_list.append(key)
            num_roles_list.append(od[key])

        # print("age_list: ", age_list)
        # print("num_roles_list: ", num_roles_list)

        pcc, pvalue = pearsonr(age_list, num_roles_list)
        print("Pearson correlation coefficient: ", pcc)
        print("P-value: ", pvalue)



class SpearmanRank:
    def __init__(self):
        print('Created SpearmanRank')



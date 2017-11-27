"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



class LRegression:
    def __init__(self, data):
        self.data = data
        print('Created Linear Regression')
        self.run_regression()

    def run_regression(self):

        # this removes the gross column from the data frame and sets it to X
        X = self.data.drop('Gross', axis=1)

        # this creates a linear regression object
        lm = LinearRegression()
        lm.fit(X, self.data.Gross)

        print('Estimated intercept coefficient: ', lm.intercept_)
        print('Number of coefficients: ', len(lm.coef_))

        # pd.DataFrame(zip(X.columns, lm.coef_), columns=['features', 'estimatedCoefficients'])
        plt.scatter(self.data.Budget, self.data.Gross)
        plt.xlabel("Total film budget USD")
        plt.ylabel("Total film gross USD")
        plt.title("Relationship between Budget and Gross Sales")
        plt.show()


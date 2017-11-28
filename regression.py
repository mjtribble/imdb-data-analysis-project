"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import model_selection


# This class runs linear regression model on the gross sales and budget of films from the imdb database using this
# tutorial http://bigdata-madesimple.com/how-to-run-linear-regression-in-python-scikit-learn/
class LRegression:
    def __init__(self, data):
        self.data = data
        print('Created Linear Regression')
        self.run_regression()

    def run_regression(self):
        # this removes the gross column from the data frame and sets it to X
        x = self.data.drop('Gross', axis=1)

        # this creates a linear regression object
        lm = LinearRegression()
        lm.fit(x, self.data.Gross)

        # plot relationship b/w budget and gross
        plt.scatter(self.data.Budget, self.data.Gross, s=10)
        plt.xlabel("Total film budget USD")
        plt.ylabel("Total film gross USD")
        plt.title("Relationship between Budget and Gross Sales")
        plt.show()

        # plot predicted and actual gross sales
        plt.scatter(self.data.Gross, lm.predict(x), s=10)
        plt.xlabel("Gross sales: $Y_i$")
        plt.ylabel("Predicted gross sales: $\hat{Y}_i$")
        plt.title("Gross sales vs Predicted gross sales $Y_i$ vs $\hat{Y}_i$")
        plt.show()

        # calculate mean squared error
        mse = np.mean((self.data.Gross - lm.predict(x)) ** 2)
        print('Mean Squared error = ', mse)

        # train data sets
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, self.data.Gross, test_size=0.33, random_state=5)
        print(x_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        # calculate mse for training and test data
        lm.fit(x_train, y_train)
        pred_train = lm.predict(x_train)
        pred_test = lm.predict(x_test)
        print("Fit a model x_train and calculate mse with y_train:",
              np.mean((y_train - pred_train) ** 2))
        print("Fit a model x_train and calculate mse with x_test, y_test:",
              np.mean((y_test - pred_test) ** 2))

        # visualize residual plot
        plt.scatter(pred_train, pred_train - y_train, c='b', s=5, alpha=0.5)
        plt.scatter(pred_test, pred_test - y_test, c='g', s=5)
        plt.hlines(y=0, xmin=0, xmax=50)
        plt.title("Residual Plot using training (blue) and test (green) data")
        plt.ylabel("Residuals")
        plt.show()

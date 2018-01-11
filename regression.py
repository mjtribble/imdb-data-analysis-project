"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats
from sklearn import model_selection


# This class runs linear regression model on the gross sales and budget of films from the imdb database using this
# tutorial http://bigdata-madesimple.com/how-to-run-linear-regression-in-python-scikit-learn/
from sklearn.metrics import accuracy_score


class LRegression:

    # This creates a new instance of the LRegression class
    # @param data = data frame of the imdb data
    def __init__(self, data):
        self.data = data
        print("\n\nQ2: As a movie's budget increases do the sales also continuously increase?")
        print('Created Linear Regression')
        self.run_regression()

    # This runs and visualizes the linear regression process
    def run_regression(self):
        # budget =  x
        x = self.data.drop('Gross', axis=1)
        x2 = np.array(x)
        x1 = []
        for i in x2:
            for j in i:
                x1.append(j)
        x1 = np.array(x1)
        y = np.array(self.data.Gross)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x1, y)

        print("slope = ", slope)
        print("r_value = ", r_value)
        print("p_value = ", p_value)
        print("standard error = ", std_err)

        # x = np.array(x)
        # y = np.array(self.data.Gross)

        # this creates a linear regression object
        lm = LinearRegression()
        lm.fit(x, self.data.Gross)

        # # calculate mean squared error
        # mse = np.mean((self.data.Gross - lm.predict(x)) ** 2)
        # print('Mean Squared error = ', mse)

        # train data sets
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, self.data.Gross, test_size=0.33, random_state=5)

        # calculate mse for training and test data
        lm.fit(x_train, y_train)
        pred_train = lm.predict(x_train)
        pred_test = lm.predict(x_test)
        print("Fit a model x_train and calculate mse with y_train:",
              np.mean((y_train - pred_train) ** 2))
        print("Fit a model x_train and calculate mse with x_test, y_test:",
              np.mean((y_test - pred_test) ** 2))

        # plot relationship b/w budget and gross
        plt.scatter(self.data.Budget, self.data.Gross, s=10)
        plt.xlabel("Total film budget (100 million USD)")
        plt.ylabel("Total film gross (100 million USD)")
        plt.title("Gross Sales per Budget (Linear Regression)")
        plt.plot(x_test, lm.predict(x_test), color='black', linewidth=3)
        plt.show()

        # plot predicted and actual gross sales
        plt.scatter(self.data.Gross, lm.predict(x), s=10)
        plt.xlabel("Actual Gross sales")
        plt.ylabel("Predicted gross sales")
        plt.title("Gross sales vs Predicted gross sales")
        plt.plot(x_test, lm.predict(x_test), color='black', linewidth=3)

        # visualize
        plt.show()

        # visualize residual plot
        plt.scatter(pred_train, pred_train - y_train, c='b', s=5, alpha=0.5)
        plt.scatter(pred_test, pred_test - y_test, c='g', s=5)
        plt.hlines(y=0, xmin=0, xmax=50)
        plt.title("Linear Regression Residual Error Plot\n"
                  "training (blue), test (green) data")
        plt.ylabel("Residuals")
        plt.show()

#
#
# class Regression:
# def __init__(self, list1, list2):
# self.list1 = list1;
# self.list2 = list2;
# print ('Linear Regression:')
# self.run()
#
#
# def run(self):
# #creates a pandas data frame which makes the data compatabel with sklearn
# X = pd.DataFrame(self.list1)
# Y = pd.DataFrame(self.list2)
#
# #this will print numerical values in the command line in respect to the scatter plot created below
# lin = linregress(self.list1, self.list2)
# print (str(lin))
#
# #creating test and train values for the model to test and train
# X_train = X[:-250]
# X_test = X[-250:]
#
# Y_train = Y[:-250]
# Y_test = Y[-250:]
#
# #This will create a 2-d scatter plot of the test data
# #It also is naming the cooridinates of the X and Y axis of the graph, overall title of the graph and the colors that the scatter plot will show
# plt.scatter(X_test,Y_test,  color='blue')
# plt.xlabel('Meta Data Score')
# plt.ylabel('IMDB Rating')
# plt.title('Relationship of MetaData and IMDB Ratings')
#
# #This method will fit the data and also show the line of regression on the scatter plot created above
# lm = linear_model.LinearRegression()
# lm.fit(X_train,Y_train)
# plt.plot(X_test, lm.predict(X_test), color='red',linewidth=3)
# plt.show()


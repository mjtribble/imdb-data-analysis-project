"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegression:
    def __init__(self, data):
        self.data = data
        print('Created Linear Regression')

import numpy as np
from collections import Counter

# ACCURACY SCORE
# get accuracy score of predictions to trues
def accuracy_score(trues, preds):

    return

# TRAIN-TEST SPLIT
# split data into
def train_test_split(x_data, y_data, train_size=0.80):

    return


# ZERO RULE
# the Zero Rule is one of the most naive baselines
# for classification, it just predicts the most class
# for regression, it predicts the average (mean or median) value
class ZeroRuleRegression():
    '''
    Zero Rule for Regression

    class-based zero-rule for regression problems

    Parameters
    ----------
    mode : str
        choose between 'mean' (default) and 'median' options

    Attributes
    -------
    '''

    def __init__(self, mode='median'):
        self.mode = mode
        self.average = 0

    # fit ("train") the function to the training data
    # inputs  : x and y data as lists or np.arrays
    # outputs : none
    def fit(self, x_data, y_data):



        return

    # predict on the test data
    # inputs : x and y data as lists or np.arrays
    # outputs : y preds as list
    def predict(self, x_data):



        return


# LINEAR REGRESSION
# assumes linear model: y = w0 + w1x1 + w2x2 + w3x3 ... (e.g. the ol' y = mx + b)
# citations
# https://www.cs.toronto.edu/~frossard/post/linear_regression/
# http://ozzieliu.com/2016/02/09/gradient-descent-tutorial/
# https://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/
class LinearRegression():
    '''
    Linear Regression with Gradient Descent

    class-based multivariate linear regression

    Parameters
    ----------
    epochs : int
        maximum epochs of gradient descent
    lr : float
        learning rate
    tol : float
        tolerance for convergence
    weights : array
        weights (coefficients) of linear model

    Attributes
    -------
    '''

    def __init__(self, epochs=100, lr=0.01, tol=1e-5):
        self.epochs=epochs
        self.lr=lr
        self.tol=tol
        self.weights = np.array([])

    # fit ("train") the function to the training data
    # inputs  : x and y data as np.arrays (x is array of x-dim arrays where x = features)
    # params  : verbose : Boolean - whether to print out detailed information
    # outputs : none
    def fit(self, x_data, y_data, verbose=False):

     

        return

    # predict on the test data
    # inputs : x data as np.array
    # outputs : y preds as list
    def predict(self, x_data):

 

        return

    # get mean squared error
    # inputs: x and y data as np.arrays
    # output: cost
    def error(self, x_data, y_data):



        return

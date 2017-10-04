import numpy as np
from collections import Counter

# ACCURACY SCORE
# get accuracy score of predictions to trues
def accuracy_score(trues, preds):

    # number of cases where pred[i] == true[i] over total number of preds
    acc = sum([1 for i in preds if preds[i]==trues[i]])/len(preds)

    return acc

# TRAIN-TEST SPLIT
# split data into
def train_test_split(x_data, y_data, train_size=0.80):

    # get index of last train example
    trainlen = int(len(y_data)*train_size)

    # shuffle indices
    order = np.random.permutation(len(y_data))

    # get portions
    x_train = x_data[order[:trainlen]]
    y_train = y_data[order[:trainlen]]
    x_test = x_data[order[trainlen:]]
    y_test = y_data[order[trainlen:]]

    return x_train, x_test, y_train, y_test


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

        # if mode is 'median', set self.average to most common
        if self.mode == 'median':

            # use Counter's most_common()
            counts = Counter(y_data)
            self.average = counts.most_common(1)[0][0]

        # otherwise, assume mode is mean
        else:

            self.average = sum(y_data)/len(y_data)

        return

    # predict on the test data
    # inputs : x and y data as lists or np.arrays
    # outputs : y preds as list
    def predict(self, x_data):

        # predict the average value for every value
        preds = [self.average for i in range(len(x_data))]

        return preds


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

        # STEP 1: ADD X_0 TERM FOR BIAS
        # so y = sum(weight*xvalue) + bias but for numpy efficiency,
        # add an 'x0' = 1.0 to our x data so we can treat bias as a weight
        x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        # STEP 2: RANDOMLY INIT WEIGHT COEFFICIENTS
        # one weight per feature (+ bias)
        # or you can use zeroes with np.zeros()
        weights = np.random.randn(x_data.shape[1])

        # STEP 3: GRADIENT DESCENT
        iters = 0
        for epoch in range(self.epochs):

            # estimate parameters, get error
            y_hat = x_data.dot(weights).flatten() # current hypothesis: y_hat = mx + b
            error = y_data.flatten() - y_hat      # how far off are we? = "loss"

            # calculate gradient and calculate new weights
            gradient = -(1.0/x_data.shape[0]) * error.dot(x_data)
            new_weights = weights - gradient * self.lr

            sqerror = np.power(error, 2)  # squared error
            # calculate cost function J
            cost = np.sum(sqerror)/2/len(y_data)

            # check stopping condition
            if np.sum(abs(new_weights - weights)) < self.tol:
                if verbose:
                    print("converged after {0} iterations".format(iters))
                break

            # update weights
            weights = new_weights
            iters += 1

            # print diagnostics
            if verbose and iters % 10 == 0:
                print("iteration {0}: cost: {1}".format(iters, cost))

        # update final weights
        self.weights = weights

        return

    # predict on the test data
    # inputs : x data as np.array
    # outputs : y preds as list
    def predict(self, x_data):

        # STEP 1: ADD X_0 TERM FOR BIAS
        x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        # STEP 2: PREDICT USING THE y_hat EQN
        preds = x_data.dot(self.weights).flatten()

        return preds

    # get mean squared error
    # inputs: x and y data as np.arrays
    # output: cost
    def error(self, x_data, y_data):

        weights = self.weights
        x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        # estimate parameters, get error
        y_hat = x_data.dot(weights).flatten()  # current hypothesis: y_hat = mx + b
        error = y_data.flatten() - y_hat  # how far off are we? = "loss"
        sqerror = np.power(error, 2)  # squared error
        # calculate cost function J
        cost = np.sum(sqerror) / 2 / len(y_data)

        return cost
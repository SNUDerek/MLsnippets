import numpy as np
from collections import Counter

# ACCURACY SCORE
# get accuracy score of predictions to trues
def accuracy_score(trues, preds):

    # number of cases where pred[i] == true[i] over total number of preds
    acc = sum([1 for i in preds if preds[i]==trues[i]])/len(preds)

    return acc

# TRAIN-TEST SPLIT
# split data into train and test sets
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
class ZeroRuleforRegression():
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

    # todo: add sgd/minibatch parameter, regularization lambda here!!!!
    def __init__(self, epochs=1000, lr=0.01, tol=1e-5):
        self.epochs=epochs
        self.lr=lr
        self.tol=tol
        self.weights = np.array([])
        self.costs_ = []

    # internal function for making hypothesis and getting cost
    def _getestimate(self, x_data, y_data, weights):

        # get hypothesis [ Andrew Ng: H_θ(x) ]
        y_hat = x_data.dot(weights).flatten()  # current hypothesis: y_hat = mx + b

        # get the difference between the trues and the hypothesis
        difference = y_data.flatten() - y_hat

        # square the difference for squared error
        squared_difference = np.power(difference, 2)

        # calculate cost function J
        # see: https://i.stack.imgur.com/tPhVh.png
        cost = np.sum(squared_difference) / 2 / len(y_data)

        return y_hat, difference, cost

    # fit ("train") the function to the training data
    # inputs  : x and y data as np.arrays (x is array of x-dim arrays where x = features)
    # params  : verbose : Boolean - whether to print out detailed information
    # outputs : cost history as list
    def fit(self, x_data, y_data, verbose=False, print_iters=100):

        # STEP 0: reset cost history
        self.costs = []

        # STEP 1: ADD X_0 TERM FOR BIAS
        # y = bias + θ_1 * x_1 + θ_2 * x_2 + ... + θ_n * x_n
        # so for p features there are p + 1 weights (all thetas + one bias)
        # add an 'x0' = 1.0 to our x data so we can treat bias as a weight
        # use numpy.hstack (horizontal stack) to add a column of ones:
        x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        # STEP 2: INIT WEIGHT COEFFICIENTS
        # one weight per feature (+ bias)
        # you can init the weights randomly:
        # weights = np.random.randn(x_data.shape[1])
        # or you can use zeroes with np.zeros():
        weights = np.zeros(x_data.shape[1])

        # STEP 3: OPTIMIZE COST FUNCTION
        # using (stochastic) gradient descent
        iters = 0

        # todo: add minibatch batch generator here!

        for epoch in range(self.epochs):

            # GRADIENT DESCENT
            # make an estimate, calculate the difference and the cost
            # then calculate gradient using cost function derivative
            # https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png
            y_hat, difference, cost = self._getestimate(x_data, y_data, weights)
            gradient = -(1.0 / x_data.shape[0]) * difference.dot(x_data)

            # todo: modify above and below for STOCHASTIC GRADIENT DESCENT

            # get new predicted weights by stepping "backwards' along gradient
            # todo: ADD REGULARIZATION HERE! (see link below)
            new_weights = weights - gradient * self.lr

            # check stopping condition
            if np.sum(abs(new_weights - weights)) < self.tol:
                if verbose:
                    print("converged after {0} iterations".format(iters))
                break

            # update weight values, save cost
            weights = new_weights
            self.costs.append(cost)
            iters += 1

            # print diagnostics
            if verbose and iters % print_iters == 0:
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

        y_hat, difference, cost = self._getestimate(x_data, y_data, weights)

        return cost


# citations
# DATASETS
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html

# LINEAR REGRESSION AND GRADIENT DESCENT
# https://www.cs.toronto.edu/~frossard/post/linear_regression/
# http://ozzieliu.com/2016/02/09/gradient-descent-tutorial/
# https://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/

# STOCHASTIC GRADIENT DESCENT
# https://www.pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/

# REGULARIZATION (FOR LINEAR REGRESSION)
# https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/

# LOGISTIC REGRESSION
# https://beckernick.github.io/logistic-regression-from-scratch/
# http://aimotion.blogspot.kr/2011/11/machine-learning-with-python-logistic.html

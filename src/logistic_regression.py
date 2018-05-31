"""
This code consists of my own Python implementation of l2 regularized
logistic regression.

Implementation by Kenten Danas for DATA 558 Statistical Machine Learning
University of Washington, June 2018
kentdanas@gmail.com
"""


import numpy as np
from scipy import stats


def obj(beta, lamda, x, y):
    """
    Function to calculate objective value for
    l2 regularized logistic regression problem
    :param beta: coefficients
    :param lamda: regularization parameter
    :param x: predictors
    :param y: response vector
    :return: objective function value
    """
    n, d = x.shape
    obj_val = (sum(np.log(1 + np.exp(-y*(x.dot(beta))))) / n) + \
              (lamda * np.linalg.norm(beta)**2)
    return obj_val


def computegrad(beta, lamda, x, y):
    """
    Function to compute the gradient for
    l2 regularized logistic regression problem
    :param beta: coefficients
    :param lamda: regularization parameter
    :param x: predictors
    :param y: response vector
    :return: gradient vector
    """
    n, d = x.shape
    y_x = y[:, np.newaxis] * x
    h = 1 + np.exp(-y_x.dot(beta))
    gradient = (np.sum(-y_x*np.exp(-y_x.dot(beta[:, np.newaxis]))/h[:, np.newaxis],axis=0)/n)\
               + (2*lamda*beta)
    return gradient


def backtracking(beta, lamda, x, y, t=1, alpha=0.5, betaparam=0.8, max_iter=100):
    """
    Function to implement backtracking line search during gradient descent.
    :param beta: coefficients
    :param lamda: regularization parameter
    :param x: predictors
    :param y: response vector
    :param t: initial step-size
    :param alpha: factor to check for sufficient decrease in objective function
    :param betaparam: decrease factor for step size update
    :param max_iter: maximum number of iterations allowed
    :return: t: updated step-size
    """
    grad_beta = computegrad(beta, lamda, x, y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_t = False
    i = 0
    while found_t is False and i < max_iter:
        # Check if chosen step size leads to sufficient decrease in objective function
        if obj(beta - t*grad_beta, lamda, x, y) < \
                obj(beta, lamda, x, y) - alpha*t*norm_grad_beta**2:
            found_t = True
        # Limit number of iterations to find optimal step size
        elif i == max_iter - 1:
            raise Exception('Maximum number of iterations of backtracking reached')
        # Decrease step size by factor of betaparam if decrease in
        # objective function is not sufficient
        else:
            t *= betaparam
            i += 1
    return t


def fastgradalgo(beta_init, lamda, x, y, t_init=1, epsilon=0.001):
    """
    Function to implement fast gradient descent ("momentum") to solve the
    l2 regularized logistic regression problem.
    :param beta_init: initial coefficients (typically zeros)
    :param lamda: regularization parameter
    :param x: predictors
    :param y: response vector
    :param t_init: initial step size for backtracking line search
    :param epsilon: stopping criteria, target accuracy for change in gradient
    :return:
        beta_vals: array of coefficients found at each iteration
        obj_vals: array of objective values found at each iteration
    """
    # Initialize coefficients and vectors to store results
    beta = beta_init
    theta = np.zeros(len(beta))
    grad_beta = computegrad(beta=theta, lamda=lamda, x=x, y=y)
    beta_vals = [beta]
    obj_vals = [obj(beta, lamda, x, y)]
    t = t_init
    i = 0

    while np.linalg.norm(grad_beta) > epsilon:
        # Find optimal step size using backtracking line search
        t = backtracking(beta=beta, lamda=lamda, x=x, y=y, t=t)

        # Perform update to coefficients
        beta_t = theta - t * grad_beta
        theta = beta_t + (i/(i+3)) * (beta_t-beta)
        beta = beta_t

        # Calculate and store new gradient using updated coefficients
        beta_vals.append(beta)
        grad_beta = computegrad(theta, lamda, x, y)

        # Calculate and store new objective value using updated coefficients
        obj_vals.append(obj(beta, lamda, x, y))

        i += 1
    return np.array(beta_vals), np.array(obj_vals)


def crossval(beta, x, y, lamdas, folds=3):
    """
    Function to perform cross-validation to find optimal regularization parameter
    :param beta: coefficients
    :param x: predictors
    :param y: response vector
    :param lamdas: list of regularization parameters to test
    :param folds: number of folds (int)
    :return: best_lamda: optimal regularization parameter
    """
    errors = []
    for lamda in lamdas:
        error = []
        for i in range(folds):
            # Randomly split data into training/test sets
            test_idx = np.random.choice(range(0, x.shape[0]), size=int(x.shape[0]/5), replace=False)
            x_test, y_test = x[test_idx], y[test_idx]
            x_train = np.delete(x, test_idx, axis=0)
            y_train = np.delete(y, test_idx, axis=0)

            # Train model
            beta_vals, obj_vals = fastgradalgo(beta_init=beta, lamda=lamda, x=x_train, y=y_train)

            # Calculate and store misclassification error
            y_hat = beta_vals[-1].dot(x_test.T)
            error.append(np.sum((y_hat - y_test)**2))
        errors.append(error)
    # Calculate average error and return lambda associated with lowest error
    avg_err = np.mean(errors, axis=1)
    best_idx = list(avg_err).index(avg_err.min())
    best_lamda = lamdas[best_idx]
    return best_lamda


def ovo(x_train, y, x_test, lamdas, t_init, classes, epsilon=0.001):
    """
    Function to use l2 regularized logistic regression for data sets with more
    than 2 classes by training one-vs-one models
    :param x_train: training predictors
    :param y: training response vector
    :param x_test: test predictors
    :param lamdas: regularization parameters to test using cross-validation
    :param t_init: initial step size for backtracking line search
    :param classes: number of classes in data set
    :param epsilon: stopping criteria, target accuracy for change in gradient
    :return: final_predictions: vector of predicted classes for test set
    """
    num_models = int(classes * (classes - 1) / 2)
    predictions = np.empty([x_test.shape[0], num_models])
    iter = 0
    for i in range(0, classes):
        for j in range(0, classes):
            if j > i:
                # Subset data to only two classes being trained on
                idx = np.where((y == i) | (y == j))
                y2 = np.array(y)[idx]
                y2[y2 == i] = -1
                y2[y2 == j] = 1
                x2 = x_train[idx]

                n, d = x2.shape
                beta2 = np.zeros(d)

                # Find optimal regularization parameter for each model using 3-fold
                # cross-validation
                best_lamda = crossval(beta2, x2, y2, lamdas, folds=3)

                # Train model on two classes
                model_betas, model_objs = fastgradalgo(beta_init=beta2, lamda=best_lamda, x=x2, y=y2,
                                                      t_init=t_init, epsilon=epsilon)
                model_beta_final = model_betas[-1]

                # Calculate and store predictions
                prediction = model_beta_final.dot(x_test.T)
                prediction[prediction >= 0] = j
                prediction[prediction < 0] = i
                predictions[:, iter] = prediction
                iter += 1

    # Get final predictions by taking most common class from one-vs-one predictions
    final_predictions = stats.mode(predictions, axis=1)[0]
    return final_predictions[:, 0]

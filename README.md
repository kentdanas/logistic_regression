# l<sub>2</sub><sup>2</sup>-regularized Logistic Regression

This repository contains my own Python implementation of l<sub>2</sub><sup>2</sup>-regularized logistic regression for solving classification problems.

I completed this code as coursework for the University of Washington's DATA 558 Statistical Machine Learning course in May 2018.

## Method Details
Implementation of l<sub>2</sub><sup>2</sup>-regularized logistic regression involves solving the convex and differentiable minimization problem:
</br>
<img src="https://github.com/kentdanas/logistic_regression/blob/master/images/logistic_regression.PNG" width=400 style='display:block; margin-left:auto; margin-right:auto'>
<br/>

Note that this assumes that the training data is of the form:
</br>
<img src="https://github.com/kentdanas/logistic_regression/blob/master/images/training_data.PNG" width=300 style='display:block; margin-left:auto; margin-right:auto'>
<br/>

Logistic regression (the loss function in the equation above) is a popular model for classification wherein the log odds of the posterior probability of the K classes is determined by a linear function of the predictor variables. l<sub>2</sub><sup>2</sup> regularization (the penalty term in the equation above) serves to prevent overfitting in the model.

## This Repository
This repository contains Python code to solve the l<sub>2</sub><sup>2</sup>-regularized logistic regression minimization problem described above. This can be used to solve classification problems for two or more classes. 

The /src folder contains my logistic_regression.py module which contains all the necessary functions to implement this method. Specifically, the module contains:
 - <b>obj</b> - function to calculate the objective value of the minimization problem described above
 - <b>computegrad</b> - function to calculate the gradient of the minimization problem described above
 - <b>backtracking</b> - function to implement backtracking line search to determine the step size for fast gradient descent (see next bullet)
 - <b>fastgradalgo</b> - function to implement the fast gradient descent ('momentum') algorithm to solve the minimzation problem described above
 - <b>crossval</b> - function to implement crossvalidation to find the optimal regularization parameter &#955;
 - <b>ovo</b> - function to implement the one-vs-one method for multi-class classification problems
 
 This respository also includes the following iPython notebooks which demonstrate the functionality of this module:
  - <b>Demo1</b> - shows an example of implementing l2 regularized logistic regression on a real-world dataset
  - <b>Demo2</b> - shows an example of implementing l2 regularized logistic regression on a simulated dataset
  - <b>Demo3</b> - compares the results of my l2 regularized logistic regression to the equivalent functions in scikit-learn on a real-world dataset

## Installation
To use the code in this repository:

 - clone the repository
 - navigate to the main directory
 - launch python
 - run "import src.logistic_regression"

Upon completing these steps all of the functions in the logisitic_regression.py module will be available. Please note that these functions were developed using Python 3.6.4 and functionality is not guaranteed for older versions of Python. If you do not already have them, you may need to install the following dependencies in your local environment:

 - matplotlob.pyplot
 - numpy
 - pandas
 - sklearn

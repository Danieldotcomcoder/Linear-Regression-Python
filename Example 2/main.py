# import library
import matplotlib.pyplot as plt
from sklearn import datasets

# Load Dataset
diabetes = datasets.load_diabetes()

# Description of the Diabetes dataset
print(diabetes.DESCR)

# Feature names
print(diabetes.feature_names)

# Create X and Y data matrices
X = diabetes.data
Y = diabetes.target
X.shape, Y.shape


# Load dataset + Create X and Y data matrices (in 1 step)
# X, Y = datasets.load_diabetes(return_X_y=True)
# X.shape, Y.shape

# import library

from sklearn.model_selection import train_test_split

# Perform 80/20 Data split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Data Dimensions

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Linear Regression Model
# Import library

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Defines the regression model
model = linear_model.LinearRegression()

# Build Training model
model.fit(X_train, Y_train)

# Apply trained model to make prediction (on test set)
Y_pred = model.predict(X_test)

# Prediction results

# Print model performance

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))

# String formatting

r2_score(Y_test, Y_pred)
r2_score(Y_test, Y_pred).dtype

# Scatter plots

# Import library

import seaborn as sns

# The Data
Y_test

import numpy as np
np.array(Y_test)

Y_pred

# Making the scatter plot
sns.scatterplot(x=Y_test, y=Y_pred)
plt.show()

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

data = pd.read_csv('advertising.csv', index_col=0)
data.head()
# add header to data frame
data.columns = ['TV', 'Radio', 'Newspaper', 'Sales']

# visualize the relation between the features and the target variable sales using scatterplots
# initialize scatterplot
fig,axs = plt.subplots(1,3,sharey=True)
# plotting Tv sales
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(16,8))
# plotting Radio Sales
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
# plotting Newspaper Sales
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])


# Applying Linear Regression analysis to estimate the relationship between sales and tv ads spending.
# lets use Tv as the feature column
feature_cols = ['TV']
# x is the independent variable
x = data[feature_cols]
# y is the dependent variable
y = data.Sales

# Importing Linear Regression model for sklearn
from sklearn.linear_model import LinearRegression
# Initializing the model
lm = LinearRegression()
# fitting the model on x and y
lm.fit(x,y)

# print the intercept and the coefficient of the resulting linear eqaution
print(lm.intercept_)
print(lm.coef_)

# lets predict the new X value
X_new = pd.DataFrame({'TV':[50]})
X_new.head()

# using this equation we will now predict the sales
lm.predict(X_new)

X_new = pd.DataFrame({'TV': [data.TV.min(),data.TV.max()] })
X_new.head()

preds = lm.predict(X_new)

#Initializing the scatter plot for tv and sales
data.plot(kind='scatter', x='TV', y='Sales')
#plotting the least squeares ling
plt.plot(X_new,preds ,c='red',linewidth=2)

# importing stat models
import statsmodels.formula.api as smf
# applying ols on tv sales
lm = smf.ols(formula='Sales ~ TV', data=data).fit()

lm.conf_int()
lm.pvalues
lm.rsquared

# lets use 3 variables to build the model
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales

from sklearn import model_selection
# create train test
xtrain,xtest,ytrain,ytest = model_selection.train_test_split(X,y,test_size=0.3,random_state=42)

# Applying Linear Regression

lm = LinearRegression()
lm.fit(X,y)
print(lm.intercept_)
print(lm.coef_)

lm = LinearRegression()
lm.fit(xtrain,ytrain)

print(lm.intercept_)
print(lm.coef_)

predictions = lm.predict(xtest)
# print the mean squared error
print(sqrt(mean_squared_error(ytest,predictions)))

lm = smf.ols(formula='Sales~TV + Radio + Newspaper', data=data).fit()
lm.conf_int()
plt.show()
print(lm.summary())


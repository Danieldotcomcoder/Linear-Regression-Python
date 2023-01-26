
# Import The Required Libraries
import pandas as pd
import numpy as np

#Import the sklearn libraries for label encoder and linear regression
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

#Import matplot library for visualization
import matplotlib.pyplot as plt




# Load the train and test datasets in pandas Dataframe
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

# Check the number of rows and columns present in train dataset
train.shape


# Print the number of columns in train dataset
train.columns

# Check the number of rows and columns present in test dataset
test.shape

# Print the number of columns in test dataset
test.columns

# combine test and train into one file to perform EDA

train["source"] = "train"
test["source"] = "test"
data = pd.concat([train,test], ignore_index = True)
print(data.shape)

# return first five elements
data.head()

# Describe function for numerical data summary

data.describe()

# checking for missing values

data.isnull().sum()



# print the unique values in the Item_fat_content column, where there are only two unique types of fat content in items: low fat and regular
data["Item_Fat_Content"].unique()

# print the unique values in the Outlet_Establishent_Year column, where the data range from 1985 to 2009
data["Outlet_Establishment_Year"].unique()

# Calculate the outlet age
data["Outlet_Age"] = 2018 - data["Outlet_Establishment_Year"]
data.head(2)

# unique values in outlet size
data["Outlet_Size"].unique()

 # print the count values in the Item_fat_content column
data["Item_Fat_Content"].value_counts()

# Print the count values of outlet sizw
data["Outlet_Size"].value_counts()

# Use the mode function to find out the most common value in the Outlet_size
data["Outlet_Size"].mode()[0]

#Two variables with missing values- Item Weight and Outlet Size
#Replacing missing values in Outlet Size with the value "medium"
data["Outlet_Size"] = data["Outlet_Size"].fillna(data["Outlet_Size"].mode()[0])

#Replacing missing values in Item weight with the mean weight
data["Item_Weight"] = data["Item_Weight"].fillna(data["Item_Weight"].mean())

# Plot a histogram to reveal the distribution of Item_Visibilty column
data["Item_Visibility"].hist(bins=20)

# Detecting Outliers
# An outlier is a data point that lies outside the overall pattern in a distribution
# A commonly used rule states that a data point is an outlier if it is more than 1.5(IQR) above the third quantile or below the first quantile
# Using this, one can remove the outliers and output the resulting data in fill-data variable.

# calculate the first quantile for Item_visibility
Q1 = data["Item_Visibility"].quantile(0.25)

# calculate the second quantile for Item_visibility
Q3 = data["Item_Visibility"].quantile(0.75)

# calulate the interquantile range (IQR)
IQR = Q3 - Q1

# now that the IQR is known, remove the outliers from the data
# The resulting data is stored in fill-data variable
fill_data = data.query('(@Q1 - 1.5 * @IQR) <= Item_Visibility <= (@Q3 + 1.5 * @IQR)')

# Display the data
fill_data.head(2)

# check the shape of the resulting dataset without the outliers
fill_data.shape

# shape of the original data
data.shape

# Assign fill data dataset to data dataframe
data = fill_data

data.shape

# Modify Item_visibility by converting the numerical values into the categories Low Visibility, Visibility, and High Visibility
data["Item_Visibility_bins"] = pd.cut(data["Item_Visibility"], [0.000, 0.065, 0.13, 0.2], labels = ['Low viz', 'Viz', 'High Viz'])

# Print the count of Item_Visibilty_bins
data["Item_Visibility_bins"].value_counts()

# Replace null values with low visibility
data["Item_Visibility_bins"] = data["Item_Visibility_bins"].replace(np.nan, "Low Viz", regex=True)

# we found typos and difference in representations in categories of Item_Fat_Content
# This can be corrected using code on screen

# Replace all other representations of low fat with Low fat
data["Item_Fat_Content"] = data["Item_Fat_Content"].replace(["low fat", "LF"], "Low Fat")

# Replace all representations of reg with Regular
data["Item_Fat_Content"] = data["Item_Fat_Content"].replace("reg", "Regular")

# print unique fat count values
data["Item_Fat_Content"].unique()

# code all categorical variables as numeric using 'LabelEncoder' from sklearn's preprocessing module
# Initialize the label encoder
le = LabelEncoder()

# Transform Item Fat_content
data["Item_Fat_Content"] = le.fit_transform(data["Item_Fat_Content"])

# Transform Item_Visibility_bins
data["Item_Visibility_bins"] = le.fit_transform(data["Item_Visibility_bins"])

# Transform Outlet_Size
data["Outlet_Size"] = le.fit_transform(data["Outlet_Size"])

# Transform Outlet_Location_Type
data["Outlet_Location_Type"] = le.fit_transform(data["Outlet_Location_Type"])

# print the unique values of outlet_type
data["Outlet_Type"].unique()

#  create dummies for outlet types
dummy = pd.get_dummies(data["Outlet_Type"])
dummy.head()

# Explore the column Item_Identifier

data["Item_Identifier"]

# As there are multiples values of food, nonconsumable items, and drinks with different numbers, combine the item type
data["Item_Identifier"].value_counts()

#As multiples categories are present, reduce my mapping

data["Item_Type_Combined"] = data["Item_Identifier"].apply(lambda x: x[0:2])
data["Item_Type_Combined"] = data["Item_Type_Combined"].map({'FD': 'Food',
                                                             'NC': 'Non-Consumable',
                                                             'Dr': 'Drinks'})


# Only three are present

data["Item_Type_Combined"].value_counts()

data.shape

# perform one hot-encoding for all columns
data = pd.get_dummies(data, columns=["Item_Fat_Content", "Outlet_Location_Type", "Outlet_Size", "Outlet_Type", "Item_Type_Combined"])

data.dtypes

import warnings
warnings.filterwarnings('ignore')

# Drop the columns which have been converted to different types

data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)
# Divide the dataset created earlier into train and test datasets

train = data.loc[data['source'] == "train"]
test = data.loc[data['source'] == "test"]

# Drop unnecessary columns

test.drop(["Item_Outlet_Sales", "source"], axis=1, inplace=True)
train.drop(["source"], axis=1, inplace=True)

# Export modified version of the files
train.to_csv("train_modified.csv", index=False)
test.to_csv("test_modified.csv", index=False)

# Read the train_modified.csv and test_modified.csv dataset
train2 = pd.read_csv("train_modified.csv")
test2 = pd.read_csv("test_modified.csv")

# print the data types of train2 columns
train2.dtypes

# Drop the irrelevant variables from train2 dataset
# Create the independent variable X-train and dependent variable Y-train
X_train = train2.drop(["Item_Outlet_Sales", "Outlet_Identifier", "Item_Identifier"],axis=1)
y_train = train2.Item_Outlet_Sales

# Drop those irrelevant variables from test2 dataset

X_test = test2.drop(["Outlet_Identifier", "Item_Identifier"], axis=1)

X_test

X_train.head(2)

y_train.head(2)

# Import sklearn libraries for model selection
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

# Create a train and test split
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# fit linear regression to the training dataset
ln = LinearRegression()

ln.fit(xtrain, ytrain)

# find the coeffiecnt and intercept
# use xtrain and ytrain for linear regression

print(ln.coef_)
ln.intercept_

# predict the results of the training data
predictions = ln.predict(xtest)
predictions

import math

# find RMSE for the model
print(math.sqrt(mean_squared_error(ytest, predictions)))

# A good RMSE for this problem is 1130, we can improve the RMSE by using algorithms like decision trees, random forest and XGboost

# Predict the column Item_Outlet_sales of test dataset
y_sales_pred = ln.predict(X_test)
y_sales_pred

test_predictions = pd.DataFrame({
    'Item_Identifier': test2['Item_Identifier'],
    'Outlet_Identifier': test2['Outlet_Identifier'],
    'Item_Outlet_Sales': y_sales_pred
} ,columns=['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']
)

test_predictions
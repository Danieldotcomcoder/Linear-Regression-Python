# importing the libraries

import numpy as np
import  matplotlib.pyplot as plt
import  pandas as pd
import  seaborn as sns



# Importing the dataset and Extracting the Independent and dependent variables

companies = pd.read_csv("1000_Companies.csv")
X = companies.iloc[:,:-1].values
Y = companies.iloc[:, 4].values

print(companies.head())

# Data visualisation
# Building the Correlation matrix
sns.heatmap(companies.corr())
# plt.show()

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder = LabelEncoder()

X[:, 3] = labelEncoder.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder="passthrough")

X = ct.fit_transform(X)
X = X[:, 1:])
X =onehotencoder.fit_transform(X.toarray())
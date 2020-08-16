# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:49:33 2020

@author: Dell
"""
# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Importing the dataset
ad_data = pd.read_csv('advertising.csv')

# Exploring the dataset
ad_data.head()

# Information and Description of dataset
ad_data.info()
ad_data.describe()

# Creating histogram of Age
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')

# Spliting the data into tarining and testing set
X=ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y=ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Training and fitting the model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Prediction and evalution
predictions = log_model.predict(X_test)
print(classification_report(y_test,predictions)) 
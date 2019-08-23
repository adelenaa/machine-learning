

#########   Simple Linear Regression  ####################

#######Importing The Libraries        ###########

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
style.use('ggplot')


################    Importing   The   Dataset     #################

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


##############  Splitting the dataset into the Training set and Test set     ################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =1/3, random_state = 0)


########### Fitting Simple Linear Regression Model To Training set    ############

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred =  regressor.predict(X_test)
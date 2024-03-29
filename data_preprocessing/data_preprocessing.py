
 ##########    importing all 3 libraries      ################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd        


###########      importing a dataset  ###################

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values  

  
###########     Splitting the dataset into Training set and Test set            ###########

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

##############   Taking care of missing values    #################
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#############       Encoding Categorical Data         ###################
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X =LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()


#############      Encoding The Dependent Variable    ###################

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y )


###########     Feature Scaling        #####################
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_tain)
y_test = sc_y.transform(y_test)
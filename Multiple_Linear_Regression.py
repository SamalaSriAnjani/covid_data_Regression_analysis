import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#predict number of antibodies present depending on age , temperature , chronic disease etc.
data_set = pd.read_csv('Covid_data Multiple Linear Regression.csv')
X = data_set.iloc[: , : -1].values
y = data_set.iloc[: , -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder ' , OneHotEncoder() , [1])] , remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.5 , random_state= 42)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)

y_pred = regressor.predict(X_test)

np.set_printoptions(precision = 3)
print(np.concatenate((y_pred.reshape(len(y_pred) , 1) , y_test.reshape(len(y_test), 1)), 1))

plt.plot(y_pred)
plt.plot(y_test)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



##### Import data ####
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("../dataset/housing_price.csv")
X, y = df.iloc[:,:12], df.iloc[:,12]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=42)


#### model training regressor
from sklearn import tree
reg = tree.DecisionTreeRegressor(max_depth=3, random_state=0)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# #### Evaluate
from sklearn.metrics import mean_absolute_error as MEA
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score
print("Mean Absolute Error:", MEA(y_test, y_pred))
print("RMSE:", RMSE(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

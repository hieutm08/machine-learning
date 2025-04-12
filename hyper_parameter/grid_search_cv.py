#### load dataset
import pandas as pd
df = pd.read_csv("../dataset/Customer-Churn-Records.csv")
df.drop(columns=["RowNumber","CustomerId","Surname","Complain","Satisfaction Score","Card Type","Point Earned"],inplace=True)

#### handle data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Geography"] = le.fit_transform(df["Geography"])
df["Gender"] = le.fit_transform(df["Gender"])

#### create the data train and test
X = df.drop(columns="Exited", axis=1)
y = df["Exited"]

#### split the data into train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=43)

#### import the model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()
knn_param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}

decision_tree_model = DecisionTreeClassifier()
decision_param_model = {'max_depth':[None, 10, 20, 30], 'min_samples_split':[2,5,10]}

knn = GridSearchCV(estimator=knn_model, param_grid=knn_param_grid, scoring="accuracy", cv=5)
dt =  GridSearchCV(estimator=decision_tree_model, param_grid=decision_param_model, scoring="accuracy", cv=5)

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
knn.fit(X_train, y_train)
dt.fit(X_train, y_train)

print("Best hyperparameter for KNN : ", knn.best_params_)
print("Best score for KNN : ", knn.best_score_)

print("Best hyperparameter for decision tree : ", dt.best_params_)
print("Best score for decision tree : ", dt.best_score_)

#### Evaluate the best model on the test data
knn_best_model = knn.best_estimator_
dt_best_model = dt.best_estimator_

knn_test_accuracy = knn_best_model.score(X_test, y_test)
dt_test_accuracy = dt_best_model.score(X_test, y_test)

print("Test accuracy for best KNN model: ", knn_test_accuracy)
print("Test accuracy for best decision tree model: ", dt_test_accuracy)
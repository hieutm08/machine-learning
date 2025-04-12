#### load dataset
import pandas as pd
df = pd.read_csv("dataset/Customer-Churn-Records.csv")
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
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_leaf=10)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, y_train, cv=10) # so lan danh gia cheo
print("cross_val_score : ",scores.mean())

from sklearn.model_selection import cross_validate

#### Evaluate
scoring = ["roc_auc","accuracy","f1"]
scores_result = cross_validate(clf, X_train, y_train, cv= 10, scoring= scoring, return_estimator=True)
for k in scores_result.keys():
    if k != "estimator":
        print(k,scores_result[k].mean())
  
testcore = []
for i in range(len(scores_result["estimator"])):
    testcore.append(scores_result["estimator"][i].score(X_test, y_test))
print("estimator scores : ",testcore)    


y_pred = scores_result["estimator"][0].predict(X_test)
print("y_pred scores : ",y_pred)

probabilities = scores_result["estimator"][0].predict_proba(X_test)
print("probabilities scores : ",probabilities)               

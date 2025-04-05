#### Load train
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('../dataset/iris_data.csv')
le = LabelEncoder()
df["class_name"] = le.fit_transform(df["class_name"])

#handle data
X, y = df.iloc[:,:4], df.iloc[:,4]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#### Load model
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=3, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#### Evaluate
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]))
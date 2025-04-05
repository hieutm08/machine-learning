#### import dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df = pd.read_csv("dataset/iris_data.csv")
df["class_name"] = le.fit_transform(df["class_name"])

#### model training
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y  = df.iloc[:,:4], df.iloc[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12, stratify=y)
# Create a KNN classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
#### Evaluate
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
print("accuracy score : ", accuracy_score(y_test,y_pred))
print("prediction score : ", roc_auc_score(y_test, knn.predict_proba(X_test), multi_class="ovr"))
print("confusion matrix : \n", confusion_matrix(y_test,y_pred))
print("blanced accuracy score : ", balanced_accuracy_score(y_test,y_pred))

from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica","Rose"]))


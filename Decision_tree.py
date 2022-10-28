import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
data = load_iris()
x = data.data
y = data.target
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30,test_size=0.45)
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Train data accuracy:",accuracy_score(y_true = y_train,y_pred = clf.predict(x_train)))
print("Test data accuracy:",accuracy_score(y_true = y_test,y_pred = y_pred))

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

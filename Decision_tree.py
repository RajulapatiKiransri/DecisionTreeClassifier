#Import necessary libraries

import pandas as pd
import numpy as np

#Dataset used for this is Iris Data
from sklearn.datasets import load_iris

#Load iris
data = load_iris()

#define the target variable and the attributes
#here, x is attibute values and y is the target variable
x = data.data
y = data.target

#divide the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30,test_size=0.45)

#the prediction classifier function
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("The predicted values are",y_pred)


#test the accuracy of the function
from sklearn.metrics import accuracy_score
print("Train data accuracy:",accuracy_score(y_true = y_train,y_pred = clf.predict(x_train)))
print("Test data accuracy:",accuracy_score(y_true = y_test,y_pred = y_pred))

#classification matrix
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

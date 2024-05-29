import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
data=pd.read_csv('D:\Ai&Ml\HeartDisease.csv')
x=data.drop('DEATH_EVENT',axis=1)
y=data['DEATH_EVENT']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=27)
log=LogisticRegression()
log.fit(x_train,y_train)
y_pred=log.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy : ",accuracy)
confusion_matrix=confusion_matrix(y_test,y_pred)
print("Confusion matrix : ",confusion_matrix)

#   OUTPUT:

#   Accuracy :  0.8166666666666667
#   Confusion matrix :  [[38  2]
#   [ 9 11]]

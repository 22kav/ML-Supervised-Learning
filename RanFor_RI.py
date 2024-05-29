import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data=pd.read_csv('D:\Ai&Ml\HeartDisease.csv')
x=data.drop('DEATH_EVENT',axis=1)
y=data['DEATH_EVENT']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=27)
ri=RandomForestClassifier()
ri.fit(x_train,y_train)
y_pred=ri.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy : ",accuracy)

#   OUTPUT:

#   Accuracy :  0.8166666666666667

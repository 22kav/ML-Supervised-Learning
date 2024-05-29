import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
data=pd.read_csv("D:\Ai&Ml\HeartDisease.csv")
x=data.drop('DEATH_EVENT',axis=1)
y=data['DEATH_EVENT']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=27)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy : ",accuracy)


#   OUTPUT:

#   Accuracy :  0.6833333333333333

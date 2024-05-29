import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
data=pd.read_csv('D:\Ai&Ml\HeartDisease.csv')
x=data.drop('DEATH_EVENT',axis=1)
y=data['DEATH_EVENT']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=27)
lin=LinearRegression()
lin.fit(x_train,y_train)
y_pred=lin.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean squared error : ",mse)
r2=r2_score(y_test,y_pred)
print("R-Squared Score : ",r2)

#   OUTPUT:

#   Mean squared error :  0.11673793021039489
#   R-Squared Score :  0.47467931405322306

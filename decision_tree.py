import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
data=pd.read_csv('D:\Ai&Ml\data_iris.csv')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=27)
#Information Gain
info_gain=DecisionTreeClassifier(criterion='entropy')
info_gain.fit(x_train,y_train)
y_pred_IG=info_gain.predict(x_test)
accuracy_IG=accuracy_score(y_test,y_pred_IG)
print("Accuracy : ",accuracy_IG)
#Gain Ratio
gain_ratio=DecisionTreeClassifier(criterion='entropy',splitter='best',max_features=None,max_depth=None,random_state=None)
gain_ratio.fit(x_train,y_train)
y_pred_GR=info_gain.predict(x_test)
accuracy_GR=accuracy_score(y_test,y_pred_GR)
print("Accuracy : ",accuracy_GR)
#Gini Index
Gini=DecisionTreeClassifier(criterion='entropy')
Gini.fit(x_train,y_train)
y_pred_gini=Gini.predict(x_test)
accuracy_gini=accuracy_score(y_test,y_pred_gini)
print("Accuracy : ",accuracy_gini)


#    OUTPUT:

#    Accuracy :  0.9
#    Accuracy :  0.9
#    Accuracy :  0.9

# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for Gradient Design.
2. Set variables for assigning dataset values.
3.Import linear regression from sklearn
4.Assign the points for representing the graph 
5. Predict the regression for marks by using the representation of the graph.

## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Gokul
RegisterNumber:  212221220013
*/
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()
X = df.iloc[:,:-1].values
X
Y = df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="yellow")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE= ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE= ",mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

### Output:
## df.head():
![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/99d90245-c5d4-4aab-9960-61c843e233ee)
## df.tail():
 ![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/c03774e8-e0fa-4e6e-8e17-a4f78ff6c836)
## Array value of X:
![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/4471d5b7-1d38-4f47-a833-fb7bae85ac08)
## Array value of Y:
![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/b333b975-5b36-4392-894c-e98738ec3fcc)
## Values of Y prediction:
![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/46239dd3-8d7f-4732-8456-37e46533f42c)
## Array values of Y test:
![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/19bad4c7-7456-4d3e-8462-8d2b54334788)
## Training set graph:
![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/6bb69b87-b4b9-45cd-89ff-43d873e8127b)
## Test set graph:
![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/b3378399-355e-40e1-a1ff-1cd6b887f552)
## Values of MSE,MAE and RMSE:

![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/dd159722-4394-4314-b812-eaa303c0c628)



















## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

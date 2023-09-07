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
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile
print("df.head():")
df.head()
print("df.tail():")
df.tail()
#Segregating data to variables
print("Array value of X:")
X=df.iloc[:,:-1].values
X
print("Array value of X:")
Y=df.iloc[:,1].values
Y
#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#displaying predicted values
print("Values of Y prediction:")
Y_pred
#displaying actual values
print("Array values of Y test:")
Y_test
#graph plot for training data
print("Training set graph:")
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#graph plot for test data
print("Test set graph:")
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("Values of MSE,MAE and RMSE:")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
print('Values of MSE')
```

## Output:
![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/2fadb635-d688-4856-897a-17683f7e951c)
![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/45eca00c-f739-417f-8220-bfea988d70b8)
![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/814827cb-bd80-4ded-9019-516e53d42502)
![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/0ece15f6-b28b-44a5-9c1d-a5bb7c878463)

![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/48199976-16b4-4108-95b1-21ceff066629)
![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/9622a585-d834-4ab9-8930-55cbc848256c)
![image](https://github.com/babavoss05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/103019882/bedb9ff5-f2f1-465e-acc0-4ea257525889)







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize weights randomly.

2.Compute predicted values.

3.Compute gradient of loss function.

4.Update weights using gradient descent.
## Program:
```

Program to implement the linear regression using gradient descent.
Developed by:vidhyasri.k
RegisterNumber:212222230170
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
                #calculate prediction
                predictions=(X).dot(theta).reshape(-1,1)
                #calculate errors
                errors=(predictions-y).reshape(-1,1)
                #update theta using gradient descent
                theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("C:/Users/admin/Downloads/50_Startups.csv")
data.head()  
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
#Learn model Parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)
#PREDICT TARGET VALUE FOR A NEW DATA POINT
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119477817/6d046f02-ca0a-4955-b9f9-68b95687f509)

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119477817/25184790-1a5a-4f57-a713-28ddc73bcce7)

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119477817/fd569092-08dc-4f10-9371-3f3b86f04024)

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119477817/76865721-7da2-4ecb-a9a5-e9f65e4b543f)

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119477817/afdb3c21-521b-4978-9ba0-9e40d8369c4a)





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

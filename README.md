# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: ALIYA SHEEMA
RegisterNumber:  212223230011
*/
```
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
```

```
dataset=pd.read_csv('Placement.csv')
print(dataset)
```

![image](https://github.com/user-attachments/assets/4f338dfa-021e-4341-a39a-958bc9a7aeea)

```
dataset.info()
```

![image](https://github.com/user-attachments/assets/e16687f3-c809-47ff-a8cb-0812d1ad7ce2)

```
dataset.drop('sl_no',axis=1,inplace=True)
```
```
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```

![image](https://github.com/user-attachments/assets/165bba28-142a-406e-9772-ddcfb130d013)

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset
```

![image](https://github.com/user-attachments/assets/34cd16f0-d35b-4468-99e9-8cecd2b4e06d)

```
dataset.info()
```

![image](https://github.com/user-attachments/assets/9f5e4da6-41b7-47d3-84fb-8b42af54920d)

```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
```
```
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,X,Y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(x.dot(theta))
    gradient=x.T.dot(h-y)/m
    theta-=alpha*gradient
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01, num_iterations=1000)
```
```
def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred
y_pred=predict(theta,X)
```
```
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
```
![image](https://github.com/user-attachments/assets/e7f80bed-667e-4084-85de-b7d990878236)

```
print(y_pred)
```

![image](https://github.com/user-attachments/assets/1150b97f-1352-42f2-8459-fbde054fd65b)

```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/443c7282-8d23-4798-9542-85f4ded46dc0)

```
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

![image](https://github.com/user-attachments/assets/36181cc3-5c44-4301-85b8-776300916225)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


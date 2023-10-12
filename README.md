# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe. 4.Apply label encoder to the non-numerical column inoreder to convert into numerical values. 5.Determine training and test data set. 6.Apply decision tree Classifier on to the dataframe 7.Get the values of accuracy and data prediction.

## Program:
```python
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: AUGUSTINE J
RegisterNumber:  212222240015
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
*/
```

## Output:
### Initial data set:
![image](https://github.com/Augustine0306/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404460/0266a6e5-67ef-4c63-b2f8-10fbf6dfe544)
### Data info:
![image](https://github.com/Augustine0306/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404460/b111462a-b1ae-4199-a31d-71b7bf044549)
### Optimization of null values:
![image](https://github.com/Augustine0306/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404460/eed18ee9-bfca-4c56-af87-0eacb8df843e)
### Assignment of x and y values:
![image](https://github.com/Augustine0306/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404460/f30dfa2e-2663-4a62-8d65-4230f3db4c59)
### Converting string literals to numerical values using label encoder:
![image](https://github.com/Augustine0306/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404460/51ea7864-7656-4c64-a6c8-40dacbdc92e1)
### Accuracy:
![image](https://github.com/Augustine0306/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404460/52f9bcc6-029e-4c8c-bb4b-41dab67a43ab)
### Prediction:
![image](https://github.com/Augustine0306/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404460/4aad4777-012f-4e43-a89c-c2f7afaee9bd)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

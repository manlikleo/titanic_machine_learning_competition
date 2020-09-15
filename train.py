import pandas as pd 
from pandas import Series,DataFrame
import numpy as np 


train = pd.read_csv("train.csv")
train


train.drop(['Cabin'],1,inplace=True)
train.dropna()


y = train['Survived']
X = train.drop(['Survived','PassengerId','Name','Ticket'],1,inplace=True)
X = pd.get_dummies(train) #convert non-numerical variables to dummy variables


from sklearn import tree
dtc = tree.DecisionTreeClassifier()
dtc.fit(X, y)
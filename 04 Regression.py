# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:08:31 2018

@author: HP
"""

import pandas as pd
import numpy as np
df = pd.read_csv("./data/apartments.csv")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['district_encoded'] = le.fit_transform(df['district'])


# Choose variables
df.drop('district', inplace=True, axis=1)

X=df.iloc[:,1:6]
y=df.iloc[:,0]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)


## Different models:
# - k-nearest neighbours regression/classification
# - Decision trees
# - Ensemble models

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

clfs = [
        KNeighborsRegressor(), 
        DecisionTreeRegressor(), 
        RandomForestRegressor(),
        LinearRegression()
        ]

max_score = 0 # Random model
best_clf=None

for clf in clfs:
    # fit the model
    clf.fit(X_train,y_train)
    
    #score
    score=clf.score(X_test,y_test)
    
    if score>max_score:
        max_score=score
        best_clf=clf
    
    
    




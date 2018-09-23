# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 12:00:28 2018

@author: HP
"""

import pandas as pd

names = ['att_'+str(i+1) for i in range(60)] + ['target']
df = pd.read_csv("./data/sonar.csv",
                 header=None,
                 index_col=None,
                 names=names)

df.head()

# Issue: target is not numeric
df['target'].unique()

# Label Encoder!
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

#fit_transform to get the encoded classes
df['target_encoded'] = le.fit_transform(df['target'])

df.iloc[0:5,-2:] #rows 0-5 and last two cols
df.loc[0:5,['target','target_encoded']]  # same thing

le.classes_
le.inverse_transform([0,0,1])

## Model time!
X = df.iloc[:,0:60]
y = df.iloc[:,-1]

# Shape of the spectra
import matplotlib.pyplot as plt
X.iloc[0,:].plot() #class 1
X.iloc[100,:].plot() # class 0
plt.legend(['Class 1','Class 0'])

# Feature selection
from sklearn.feature_selection import SelectKBest

fs=SelectKBest(k=10)
fs.fit(X,y)
X_fs=fs.transform(X) 

# Now I can use only X_fs...
# Same workflow can follow from here (using X_fs 
# instead of X): train_test_split, 
# model selection, etc. 

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    random_state=123)

pipe = make_pipeline(
        SelectKBest(k=10),
        MinMaxScaler(),
        LogisticRegression()
        )

pipe.fit(X_train,y_train)
pipe.score(X_test,y_test)

# Cross validation score:
import numpy as np
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X_test,y_test, cv=10)
print("Average R2", np.mean(scores)) # Mean R2 score
print("Std of R2", np.std(scores)) # Std of R2



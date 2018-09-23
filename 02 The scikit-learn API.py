# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 09:53:58 2018

@author: HP
"""
###########################################
#
# Classification / Discrete indep variable
#
###########################################


from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Our goal: find a model that estimates
# flower class from the 4 measurements

# 1. Train-test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    test_size=0.2,
                                                    random_state=42)

# 2. Make a model

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train, y_train)
clf.coef_ # Coefficients of the model

# 3. See predictions on the test data
y_pred = clf.predict(X_test)

# How many did we get wrong?
sum(y_pred != y_test)

# For classification problems:
# we can use a classification report

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

###########################################
#
# Regression / Continuous indep variable
#
###########################################

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_boston(return_X_y=True)

### YOUR TASK:
# 1. Train test split
# 2. Find linear regression model
# 3. Measure performance
# BONUS 1: add scaler 
# BONUS 2: Identify the best predictors

## SOLUTION:
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

###########################
# Vanilla linear regression
###########################

clf = LinearRegression()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# R2 error of the model
clf.score(X_test, y_test)

clf.coef_

# Cross-validation score: measure
# performance across differen train/test splits
import numpy as np
from sklearn.model_selection import cross_val_score
cv_folds_score = cross_val_score(clf, X_test, y_test, cv=5)
print("R2 score: {} with std {}".format(
        np.mean(cv_folds_score), np.std(cv_folds_score))
)

###########################
# Pipelines
###########################

pipe = make_pipeline(StandardScaler(), 
                      LinearRegression()
                      )

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

pipe.score(X_test,y_test)
###########################
# Parity plot: Model diagnosis
###########################

# 45 degree line
import numpy as np
import matplotlib.pyplot as plt
idx = np.arange(0,50)
plt.scatter(y_test,y_pred)
plt.plot(idx,idx)
plt.xlabel("True value")
plt.ylabel("Predicted value")

# Feature importance
coefs = clf.coef_ 
n_coefs = len(coefs)
pos = range(n_coefs)
names = ['att_'+str(i+1) for i in range(n_coefs)]
plt.barh(pos,coefs)
plt.yticks(pos,names)









# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 22:34:19 2021

@author: Sakshi Tayade
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine Learning Library
# Function for splitting dataset and cross validation scores on training set
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

# Function to generate the output
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Function for Algorithms implementation
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

names = ['sepal-lenght' , 'sepal-width' , 'petal-length' , 'petal-width' , 'class']
ds = pd.read_csv(r'D:\Apan ka Python\Python_DataScience_with_Shrikant_pande_sir\NOTES\iris.csv' , names=names)

array = ds.values #convert entire dataset into the array
print(array)
x = array[:,0:4]  # input variabls sl(0) sw(1) pl(2) pw(3)
y = array[:,4]    # output variables "class"

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.20 , random_state = 1)

results = []    # created an empty list to store results of the algorithms
names = []     # created an empty list to stores names of the algorithms
mean = []
models = []
# Step 6 :- Make the prediction on the Testing Set

model = DecisionTreeClassifier() 
model.fit(x_train , y_train)
predictions = model.predict(x_test)
print(x_test)
print(predictions)
print('DT : on Testing Set ' , accuracy_score(y_test , predictions)*100)

for name , model in models :
    kfold = StratifiedKFold(n_splits = 10 , random_state = 1 , shuffle = True)
    cv_results = cross_val_score(model , x_train , y_train , cv = kfold , scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    mean.append(cv_results.mean())
    
    print('%s : %f (%f)' % (name , cv_results.mean() , cv_results.std()))
colors = ['grey' , 'blue' , 'red' , 'green' , 'yellow']
plt.bar(names,mean)
plt.title('Algorithm Comparison')
plt.show()

plt.boxplot(results)
plt.title('Algorithm')
plt.show()

# Step 7 :- Final Output 

print(accuracy_score(y_test , predictions))
print(confusion_matrix(y_test , predictions))
print(classification_report(y_test , predictions))
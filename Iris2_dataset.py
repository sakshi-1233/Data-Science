# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 22:00:15 2021

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

# Step 5:- Supervised Machine Learning
# Split the data into two parts 1) 4 variables 2) class variable

array = ds.values #convert entire dataset into the array
print(array)
x = array[:,0:4]  # input variabls sl(0) sw(1) pl(2) pw(3)
y = array[:,4]    # output variables "class"

print("Values of x----->")
print()
print(x)
print()
print("values of y ----->")
print(y)
print()

# x = fist 4 cols with 600 records
# y = las col or class with all 600 records

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.20 , random_state = 1)

# x_train       first 4 cols 80%     480 lines
# x_test        first 4 cols 20-%    120 lines
# y_train       last col 80%         480 lines
# y_test        last col 20%         120 lines

#model = DecisionTreeClassifier()
#kfold = StratifiedKFold(n_splits = 10 , random_state = 1 , shuffle = True)
#cv_results = cross_val_score(model,x_train , y_train , cv = kfold , scoring = 'accuracy')
#print('DT :- ' , cv_results.mean())
#print(cv_results)  #10

#model = GaussianNB()
#kfold = StratifiedKFold(n_splits = 10 , random_state = 1 , shuffle = True)
#cv_results = cross_val_score(model,x_train , y_train , cv = kfold , scoring = 'accuracy')
#print('GaussianNB :- ' , cv_results.mean())
#print(cv_results)

#model = KNeighborsClassifier()
#kfold = StratifiedKFold(n_splits = 10 , random_state = 1 , shuffle = True)
#cv_results = cross_val_score(model,x_train , y_train , cv = kfold , scoring = 'accuracy')
#print('KNeighbour :- ' , cv_results.mean())
#print(cv_results)

# Anmother way to use all the algorithms
# Step 6 :- Make the prediction on the Testing Set

models = []  #creating an empty list
models.append(('LR' , LogisticRegression(solver='liblinear' , multi_class = 'ovr')))
models.append(('KN' , KNeighborsClassifier()))
models.append(('DT' , DecisionTreeClassifier()))
models.append(('SNM' , SVC(gamma = 'auto')))
models.append(('NB' , GaussianNB()))

# models list would be created with 5 algorithms functions
results = []    # created an empty list to store results of the algorithms
names = []     # created an empty list to stores names of the algorithms
mean = []

for name , model in models :
    kfold = StratifiedKFold(n_splits = 10 , random_state = 1 , shuffle = True)
    cv_results = cross_val_score(model , x_train , y_train , cv = kfold , scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    mean.append(cv_results.mean())
    
    print('%s : %f (%f)' % (name , cv_results.mean() , cv_results.std()))
colors = ['grey' , 'blue' , 'red' , 'green' , 'yellow']
plt.bar(names,mean, color = colors)
plt.title('Algorithm Comparison')
plt.show()

plt.boxplot(results , labels = names)
plt.title('Algorithm Comparison')
plt.show()

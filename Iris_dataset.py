import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Step 1:- Understanding the Dataset

names = ['sepal-lenght' , 'sepal-width' , 'petal-length' , 'petal-width' , 'class']
ds = pd.read_csv(r'D:\Apan ka Python\Python_DataScience_with_Shrikant_pande_sir\NOTES\iris.csv' , names=names)

print(ds.shape)
print("First 10  Data --------->")
print()
print(ds.head(10))
print()
print("Last 10 Data--------->")
print()
print(ds.tail(10))
print()

# Step 2 :- Statistical Summary of DataSet
print("Describe() Function ::-")
print(ds.describe())
print()

#Step 3:- Occurence of Each Class
print(ds['class'].value_counts())
print()

# Step 4:- Visualize data by graphs
# univariate plot - Boxplot 
ds.plot(kind='box',subplots=True , layout=(2,2))
plt.suptitle("Boxplot on the variables of iris")
plt.show()
ds.hist()
plt.show()
print()

# Multivariate plot 
#Pairplot
sns.pairplot(ds , hue='class' , markers = '+')
plt.show()
# Boxplot
plt.figure(figsize = (15,10))
plt.subplot(2,2,1)
sns.boxplot(x='class',y='sepal-lenght',data= ds)
plt.subplot(2,2,2)
sns.boxplot(x='class',y ='sepal-width',data= ds)
plt.subplot(2,2,3)
sns.boxplot(x='class',y ='petal-length',data= ds)
plt.subplot(2,2,4)
sns.boxplot(x='class',y ='petal-width',data= ds)
plt.show()

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

model = DecisionTreeClassifier()
kfold = StratifiedKFold(n_splits = 10 , random_state = 1 , shuffle = True)
cv_results = cross_val_score(model,x_train , y_train , cv = kfold , scoring = 'accuracy')
print(cv_results.mean())

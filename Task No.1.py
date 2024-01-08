#!/usr/bin/env python
# coding: utf-8

# # <font color=purple>Oasis Infobyte Internship</font>
# Intern Name -Akshay Anandkar

# # Task 1-IRIS FLOWER CLASSIFICATION 
Problem Statement-Iris flower has three species; setosa, versicolor, and virginica, which differs according to their measurements. Now assume that you have the measurements of the iris flowers according to their species, and here your task is to train a machine learning model that can learn from the measurements of the iris species and classify them.


# In[1]:


#import all require liabraries
import pandas as pd
import numpy as np


# In[2]:


#now importing iris dataset 
iris=pd.read_csv(r"D:\Data-Science-Internship\Iris.csv")
iris.head(10)


# In[3]:


#checking dataset
iris.info()


# In[4]:


#Checking for null values
iris.isnull().sum()


# In[5]:


#Taking independent varaible
X=iris.drop(['Species'],axis=1)
X


# In[6]:


#dependent varaible
y=iris['Species']


# In[7]:


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.02,random_state=0)


# In[8]:


print(X_train)


# In[9]:


#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[10]:


#Applying decision tress classifier model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)


# In[11]:


X_test=sc.transform(X_test)


# In[12]:


#Gettig y_prediction
y_pred=classifier.predict(X_test)
y_pred


# In[13]:


classifier.score(X_train,y_train)


# In[14]:


classifier.score(X_test,y_test)


# In[15]:


#Applying Classification report
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[16]:


cr=classification_report(y_test,y_pred)
print(cr)







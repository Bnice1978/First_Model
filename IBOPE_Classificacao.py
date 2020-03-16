#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


# Reading the data
data = pd.read_csv("IBOPE_Cluster_new.csv",sep=',')
data_copy = data.drop(['Municipio'], axis = 1)

# looking at the first five rows of the data
data_copy.head()


# In[3]:


# Organize the data
target = data_copy['Cluster']
features = data_copy.drop(['Cluster'], axis = 1)

# looking at the first five rows of the data
features.head()


# In[4]:


# Create train and test set
x_train, x_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.25,
                                                    random_state=0)


# In[5]:


# Estimate the K-nearest neighbor model
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)

knn.score(x_test, y_test)


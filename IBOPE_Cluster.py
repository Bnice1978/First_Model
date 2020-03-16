#!/usr/bin/env python
# coding: utf-8

# In[26]:


# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans


# In[27]:


# Reading the data
data = pd.read_csv("IBOPE.csv",sep=';')
data_copy = data.drop(['Municipio'], axis = 1)

#looking at the first five rows of the data
data_copy.head()


# In[28]:


# Statistics of the data
data_copy.describe()


# In[29]:


# Standardizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_copy)

# Statistics of scaled data
pd.DataFrame(data_scaled).describe()


# In[30]:


# Fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# Converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[31]:


# Defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=600)

# Fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)


# In[32]:


# Inertia on the fitted data
kmeans.inertia_


# In[33]:


pred = kmeans.predict(data_scaled)
data['Cluster'] = pred
data['Cluster'].value_counts()


# In[34]:


# Export output to csv
data.to_csv (r'IBOPE_Cluster_new.csv', index = False, header=True)


# In[49]:


# Statistics of data grouped by cluster
data_describe = data.groupby('Cluster').describe()

# Export output to csv
data_describe.to_csv (r'IBOPE_Cluster_Describe.csv', index = False, header=True)


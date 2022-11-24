#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#hierarchical Clustering


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[2]:


data = pd.read_csv("crime_data.csv")


# In[3]:


data.head()


# In[4]:


data.dtypes


# In[5]:


data.isnull().sum()


# In[6]:


data1 = data.rename(columns={"Unnamed: 0":"USA"})
data1.head()


# In[7]:


data1.duplicated().value_counts()


# In[ ]:


#step1-normalize the data


# In[8]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)
df_norm = norm_func(data1.iloc[:,1:])
df_norm.head()


# In[ ]:


#step 2 - Dendrogram


# In[9]:


dendrogram = sch.dendrogram(sch.linkage(df_norm,method="complete"))


# In[ ]:


#step 3 create clusters 


# In[10]:


hc = AgglomerativeClustering(n_clusters = 4,affinity='euclidean',linkage='complete')
df_pred =hc.fit_predict(df_norm)
df_pred


# In[11]:


hc_data =data1.copy()
hc_data.head()


# In[12]:


hc_data["Clusters"] = df_pred
hc_data.head()


# In[15]:


hc_data.iloc[:,1:5].groupby(hc_data.Clusters).mean()


# In[ ]:


#Method 2 - kmeans


# In[18]:


from sklearn.cluster import KMeans
data1.head()


# In[ ]:


#ELBOW CURVE
FOR IDENTIFICATION OF HOW MANY CLUSTER SHOULD PERFORM


# In[21]:


wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("Elbow Curve")
plt.xlabel("n_numbers of clusters")
plt.ylabel("wcss")


# In[ ]:


#Crete KMeans Model


# In[22]:


model = KMeans(n_clusters=4)
model.fit(df_norm)
model.labels_


# In[23]:


km_data = data1.copy()
km_data.head()
km_data["km_Cluster"] = model.labels_
km_data.iloc[:,0:5].groupby(km_data["km_Cluster"]).mean()


# In[ ]:


#Method 3 : DBSCAN


# In[24]:


from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


# In[25]:


data1.head()


# In[27]:


db = data1.iloc[:,1:5].values
sc = StandardScaler().fit(db)
x = sc.transform(db)


# In[28]:


dbscan =DBSCAN(eps=2,min_samples=6)
dbscan.fit(x)
dbscan.labels_


# In[29]:


cl = pd.DataFrame(dbscan.labels_,columns=["clust"])
cl.head()


# In[30]:


data1["clust"] = cl
data1.head()


# In[ ]:





# In[ ]:





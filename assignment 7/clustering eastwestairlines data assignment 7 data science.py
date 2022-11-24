#!/usr/bin/env python
# coding: utf-8

# In[ ]:


perform clustering (hierarchical,K means clustering and DBSCAN) 
for the airlines data to obtain optimum number of clusters.
Draw the inferences from the clusters obtained.


# In[ ]:


#Method 1 : DBSCAN


# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("EastWestAirlines.csv")
data.head()


# In[2]:


data.dtypes


# In[3]:


data.isnull().sum()


# In[4]:


data.duplicated().value_counts()


# In[ ]:


#standardize the dataset


# In[5]:


df = data.values
sc =StandardScaler().fit(df)
x =sc.transform(df)
x


# In[ ]:


#apply DBSCAN


# In[7]:


dbscan = DBSCAN(eps=2,min_samples=13)
dbscan.fit(x)
dbscan.labels_


# In[8]:


data1 = data.copy()
data1["cluster"] = dbscan.labels_
data1.groupby(data1["cluster"]).mean()


# In[9]:


data1["cluster"].value_counts()


# In[16]:


for i in range(0,12):
    sns.barplot(y=data1.iloc[:,i],x=data1["cluster"])
    plt.xlabel("DBSCAN Clusters")
    plt.ylabel(data1.columns[i])
    plt.title(f"DBSCAN Clustering")
    plt.show()


# In[ ]:


#Method2: KMeans Clustering


# In[17]:


from sklearn.cluster import KMeans
data2 = data.copy()
data2.head()


# In[ ]:


#normalize the data


# In[18]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)
df_norm = norm_func(data2)
df_norm


# In[ ]:


#elbow curve


# In[20]:


wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("Elbow Curve")
plt.xlabel("n_numbers of clusters")
plt.ylabel("wcss")


# In[21]:


kmeans = KMeans(n_clusters=5)
kmeans.fit(df_norm)
kmeans.labels_


# In[22]:


data2["clusters"] = kmeans.labels_
data2.groupby(data2['clusters']).mean()


# In[23]:


data2['clusters'].value_counts()


# In[ ]:


#hierarchy clustering


# In[24]:


from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
data3 = data2.copy()
data3.info()


# In[ ]:


#Dendrogram


# In[25]:


dendrogram = sch.dendrogram(sch.linkage(df_norm,method="complete"))


# In[26]:


hc = AgglomerativeClustering(n_clusters = 5,affinity='euclidean',linkage='complete')
df_pred = hc.fit_predict(df_norm)
df_pred


# In[27]:


data3['cluster'] = df_pred
data3
data3.groupby(data3['cluster']).mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





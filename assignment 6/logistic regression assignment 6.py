#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Output variable -> y

y -> Whether the client has subscribed a term deposit or not

Binomial ("yes" or "no")


# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression


# In[8]:


data=pd.read_csv("bank-full.csv",sep=';')


# In[9]:


data.describe()


# In[10]:


data.dtypes


# In[11]:


data.isnull().sum()


# In[12]:


data.shape


# In[13]:


data.duplicated().value_counts()


# In[ ]:


#dividing data into input and output


# In[14]:


y=data.iloc[:,[-1]]
y.head()


# In[16]:


sns.countplot(data["y"])


# In[18]:


x=data.iloc[:,0:15]
x.head()


# In[20]:


from sklearn import preprocessing
le= preprocessing.LabelEncoder()
x1 =x.apply(le.fit_transform)
x1.head()


# In[21]:


classifier =LogisticRegression()
classifier.fit(x1,y)


# In[ ]:


#for predictions


# In[22]:


y_pred = classifier.predict_proba(x1)
y_pred


# In[23]:


y_pred = classifier.predict(x1)


# In[24]:


y_pred


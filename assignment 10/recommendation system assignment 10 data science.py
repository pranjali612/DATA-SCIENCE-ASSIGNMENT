#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
books = pd.read_csv("book (1).csv",encoding='Latin1')


# In[2]:


books.head()


# In[3]:


books_df=books.iloc[:,1:]
books_df


# In[4]:


books_df.sort_values('User.ID')


# In[5]:


len(books_df['User.ID'].unique())


# In[6]:


len(books_df['Book.Title'].unique())


# In[7]:


books_df['Book.Title'].value_counts()


# In[8]:


user_books_df = books_df.pivot_table(index='User.ID',columns='Book.Title',values='Book.Rating')
user_books_df


# In[9]:


# Converting Nan into 0
user_books_df.fillna(0,inplace=True)
user_books_df


# In[10]:


#calculating Cosine similarly between users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation


# In[11]:


user_sim = 1-pairwise_distances(user_books_df.values,metric='cosine')
user_sim


# In[12]:


user_sim_df = pd.DataFrame(user_sim)


# In[13]:


#Set the index and column names to user ids
user_sim_df.index=books_df['User.ID'].unique()
user_sim_df.columns=books_df['User.ID'].unique()


# In[14]:


#Nullifying diagonal values
np.fill_diagonal(user_sim,0)
user_sim_df.iloc[0:5,0:5]


# In[15]:


# Finding Similar Users
user_sim_df.idxmax(axis=1)


# In[17]:


books_df[(books_df['User.ID']==162121)|(books_df['User.ID']==276726)]


# In[20]:


user_1 = books_df[(books_df['User.ID']==162121)]
user_2 = books_df[(books_df['User.ID']==276726)]
user_1['Book.Title']


# In[21]:


user_2['Book.Title']


# In[22]:


pd.merge(user_1,user_2,on='Book.Title',how='outer')


# In[ ]:





# In[ ]:





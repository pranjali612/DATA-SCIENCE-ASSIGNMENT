#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
zoo_data = pd.read_csv("Zoo.csv")
zoo_data
zoo_data.info()


# In[2]:


zoo_data.dtypes


# In[3]:


zoo_data.isnull().sum()


# In[4]:


zoo_data.duplicated().sum()


# In[5]:


zoo_data.describe()


# In[6]:


zoo_data['type'].unique()


# In[7]:


plt.figure(figsize=(10,6))
sns.countplot(zoo_data['type'])
plt.title('Count Plot')
plt.grid(True)
plt.show()


# In[9]:


zoo_data.drop('animal name',axis=1,inplace=True)


# In[10]:


zoo_data.head()


# In[11]:


X=zoo_data.drop('type',axis=1)
y=zoo_data[['type']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=10)
print('X_train_shape :',X_train.shape , '\ny_train_shape :',y_train.shape)
print('X_test_shape :',X_test.shape , '\ny_test_shape :',y_test.shape)


# In[12]:


model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train,y_train)


# In[13]:


pred_y= model.predict(X_train)


# In[14]:


accuracy_score(y_train,pred_y)


# In[15]:


confusion_matrix(y_train,pred_y)


# In[ ]:





# In[16]:


print(classification_report(y_train,pred_y))


# In[17]:


y_pred=model.predict(X_test)


# In[18]:


#accuracy score for test data
accuracy_score(y_test,y_pred)


# In[19]:


#confusion Matrix
confusion_matrix(y_test,y_pred)


# In[20]:


print(classification_report(y_test,y_pred))


# In[21]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 41
k_range = range(1, 41)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5)
    k_scores.append(scores.mean())


# In[22]:


# plot to see clearly
plt.figure(figsize=(10,6))
plt.plot(k_range, k_scores,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.grid(True)
plt.title('CV Accuracy Vs k value for KNN')
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





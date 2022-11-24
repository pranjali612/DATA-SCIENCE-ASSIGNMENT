#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[3]:


df = pd.read_csv('glass.csv')


# In[4]:


df.head()


# In[12]:


X = np.array(df.iloc[:,3:5])
y = np.array(df['Type'])
print("shape of X:"+str(x.shape))
print("shape of y:"+str(y.shape))


# In[7]:


cm_dark = ListedColormap(['#ff6060', '#8282ff','#ffaa00','#fff244','#4df9b9','#76e8fc','#3ad628'])
cm_bright = ListedColormap(['#ffafaf', '#c6c6ff','#ffaa00','#ffe2a8','#bfffe7','#c9f7ff','#9eff93'])


# In[13]:


plt.scatter(X[:,0],x[:,1],c=y,cmap=cm_dark,s=10,label=y)
plt.show()


# In[10]:


sns.swarmplot(x='Na',y='RI',data=df,hue='Type')


# In[14]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y)
print("Shape of X_Train:"+str(X_train.shape))
print("Shape of y_Train:"+str(Y_train.shape))
print("Shape of X_Test:"+str(X_test.shape))
print("Shape of y_Test:"+str(Y_test.shape))


# In[15]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)


# In[16]:


pred = knn.predict(X_train)
pred


# In[17]:


accuracy = knn.score(X_train,Y_train)
print("The accuracy is :"+str(accuracy))


# In[18]:


cnf_matrix = confusion_matrix(Y_train,pred)
print(cnf_matrix)


# In[19]:


plt.imshow(cnf_matrix,cmap=plt.cm.jet)


# In[20]:


df_cm = pd.DataFrame(cnf_matrix, range(6),range(6))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 16})


# In[21]:


h = .02  
n_neighbors = 5
for weights in ['uniform', 'distance']:
    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cm_bright)


plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_dark,
                edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i, weights = '%s')"% (n_neighbors, weights))

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





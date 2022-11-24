#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[2]:


forest=pd.read_csv("forestfires.csv")


# In[3]:


forest


# In[4]:


forest.info()


# In[5]:


forest.describe()


# In[6]:


forest.isnull().sum()


# In[7]:


forest.corr()


# In[ ]:


sns.pairplot(forest)


# In[8]:


# Dropping columns which are not required

data = forest.drop(['dayfri', 'daymon', 'daysat', 'daysun', 'daythu','daytue', 'daywed', 'monthapr', 'monthaug', 'monthdec', 
                  'monthfeb','monthjan', 'monthjul', 'monthjun', 'monthmar', 'monthmay', 'monthnov','monthoct','monthsep','month','day'], 
                 axis = 1)
data


# In[9]:


# Checking how much datapoints are having small and large area
data.size_category.value_counts()


# In[10]:


sns.countplot(x = 'size_category', data = data)


# In[11]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data['size_category']= label_encoder.fit_transform(data['size_category'])
data


# In[12]:


import matplotlib.pyplot as plt
plt.figure(figsize = (18,12))
sns.heatmap(data.corr(), annot=True, cmap="YlOrRd")
ax = plt.gca()
ax.set_title("HeatMap of Features for the Classes")


# In[13]:


array = data.values
x = array[:,0:-1]
y = array[:,-1]


# In[14]:


x


# In[15]:


y


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)


# In[ ]:


#Grid Search CV


# In[17]:


clf = SVC()
param_grid = [{'kernel':['linear', 'poly', 'rbf', 'sigmoid'],'gamma':[50,5,10,0.5,0.1,10],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=5)
gsv.fit(x_train,y_train)


# In[18]:


gsv.best_params_ , gsv.best_score_


# In[19]:


clf = SVC(kernel="linear",C = 15, gamma = 50)
clf.fit(x_train , y_train)
y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[ ]:


#Prepare a classification model using SVM for salary data


# In[21]:


salary_test=pd.read_csv("C:/Users/pranjali/Downloads/SalaryData_Test.csv")


# In[22]:


salary_test


# In[23]:


salary_train=pd.read_csv("C:/Users/pranjali/Downloads/SalaryData_Train.csv")


# In[24]:


salary_train


# In[25]:


salary_test.info()


# In[26]:


salary_train.info()


# In[27]:


salary_test.describe()


# In[28]:


salary_train.describe()


# In[29]:


salary_test.isnull().sum()


# In[30]:


salary_train.isnull().sum()


# In[31]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[32]:


lst = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]


# In[33]:


for i in lst:
     salary_train[i]= lb.fit_transform(salary_train[i])


# In[34]:


for i in lst:
     salary_test[i]= lb.fit_transform(salary_test[i])


# In[35]:


salary_test


# In[36]:


salary_train


# In[37]:


X_train=salary_train.iloc[:,0:-1]
Y_train=salary_train['Salary']


# In[38]:


Y_train


# In[39]:


X_test=salary_test.iloc[:,:-1]
Y_test=salary_test['Salary']


# In[40]:


X_test


# In[41]:


Y_test


# In[ ]:


# svm with kernel rbf


# In[42]:


clf = SVC(kernel="rbf")
clf.fit(X_train , Y_train)
Y_pred = clf.predict(X_test)
acc = accuracy_score(Y_test, Y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(Y_test, Y_pred)


# In[ ]:


# svm with kernel sigmoid


# In[43]:


clf = SVC(kernel="sigmoid")
clf.fit(X_train , Y_train)
Y_pred = clf.predict(X_test)
acc = accuracy_score(Y_test, Y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(Y_test, Y_pred)


# In[ ]:


# svm with kernel poly


# In[44]:


clf = SVC(kernel="poly")
clf.fit(X_train , Y_train)
Y_pred = clf.predict(X_test)
acc = accuracy_score(Y_test, Y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(Y_test, Y_pred)


# In[ ]:





# In[ ]:





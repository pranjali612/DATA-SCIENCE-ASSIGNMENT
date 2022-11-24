#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Use Random Forest to prepare a model on fraud data


# In[ ]:


#importing data


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix


# In[3]:


fraud_data = pd.read_csv('Fraud_check.csv')
fraud_data


# In[4]:


fraud_data.info()


# In[5]:


fraud_data.isnull().sum()


# In[6]:


fraud_data.duplicated().sum()


# In[7]:


fraud_data.describe(include='all')


# In[13]:


corr=fraud_data.corr()
corr


# In[14]:


sns.heatmap(corr,annot=True)
plt.show()


# In[15]:


fraud_data


# In[16]:


sns.countplot(x='Marital.Status',data=fraud_data)
plt.grid(True)
plt.show()

sns.countplot(x='Urban',data=fraud_data)
plt.grid(True)
plt.show()

sns.countplot(x='Undergrad',data=fraud_data)
plt.grid(True)
plt.show()


# In[18]:


print('minimum_value : ' , fraud_data['Taxable.Income'].min() ,'\n maximun_value :',fraud_data['Taxable.Income'].max())


# In[19]:


#Converting Target variable 'Sales' into categories Low, Medium and High.
fraud_data['Taxable.Income'] = pd.cut(x=fraud_data['Taxable.Income'],bins = [10002,30000,99620], labels=['Risky','Good'])
fraud_data['Taxable.Income']


# In[20]:


fraud_data.head()


# In[21]:


sns.countplot(fraud_data['Taxable.Income'])
plt.grid(True)
plt.show()


# In[22]:


fraud_data['Taxable.Income'].value_counts()


# In[23]:


fraud_data = pd.get_dummies(fraud_data,columns = ["Taxable.Income"],drop_first=True)


# In[24]:


fraud_data.head()


# In[25]:


#subscription to term deposit
plt.figure(figsize=(10,6))
plt.pie(fraud_data['Taxable.Income_Good'].value_counts(),labels=['Good','Risky'],explode=(0,0.1),autopct ='%1.2f%%')
plt.title('Pie Chart')
plt.show()


# In[ ]:


by piechart we say that our data is imbalanced. We have to take same percentage of 'Good' & 'Risky' in training
and testing data.


# In[26]:


#encoding categorical fraud_data
label_encoder = preprocessing.LabelEncoder()

fraud_data['Undergrad'] = label_encoder.fit_transform(fraud_data['Undergrad'])
fraud_data['Taxable.Income_Good'] = label_encoder.fit_transform(fraud_data['Taxable.Income_Good'])
fraud_data['Marital.Status'] = label_encoder.fit_transform(fraud_data['Marital.Status'])
fraud_data['Urban'] = label_encoder.fit_transform(fraud_data['Urban'])

fraud_data


# In[ ]:


#Data Preparation


# In[27]:


x=fraud_data.drop('Taxable.Income_Good',axis=1)
y=fraud_data[['Taxable.Income_Good']]


# In[ ]:


Spliting and handling imbalanced data


# In[29]:


# by using stratify=y,we can deal with imbalanced data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 12,shuffle=True,stratify=y)


# In[30]:


plt.figure(figsize=(10,6))
#train data 
ax1 = plt.subplot(121)
line1=plt.pie(y_train.value_counts(),labels=['Good','Risky'],explode=(0,0.1),autopct ='%1.2f%%')
plt.title('Pie Chart train data')

#test data
ax2 = plt.subplot(122)
line2=plt.pie(y_test.value_counts(),labels=['Good','Risky'],explode=(0,0.1),autopct ='%1.2f%%')
plt.title('Pie Chart test data')
plt.show()


# In[ ]:


by observing above plots, we can say that in our training and testing data the percentage of 'Good' & 'Risky' are same 
as original dataset


# In[35]:


print('X_train_shape :',x_train.shape,'\ny_train_shape',y_train.shape)


# In[37]:


print('X_test_shape :',x_test.shape,'\ny_test_shape',y_test.shape)


# In[ ]:


#Model Building


# In[38]:


rf_classifier = RandomForestClassifier(random_state=38)
rf_classifier.fit(x_train,y_train)


# In[ ]:


#Grid SearchCv


# In[ ]:


#To check which criterion is best for our RandomForest Classifier and also which Max_depth is best for our
RandomForest Classifier.


# In[41]:


from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator = rf_classifier,
                           param_grid = {'criterion':['entropy','gini'],
                                         'max_depth':[2,3,4,5,6,7,8,9,10]},
                           cv=5)
grid_search.fit(x,y)
print(grid_search.best_params_)
print(grid_search.best_score_)


# In[43]:


#new Model
rf_classifier_1 = RandomForestClassifier(criterion = 'entropy',random_state=38,max_depth=2)
rf_classifier_1.fit(x_train, y_train)


# In[44]:


y_pred_train =rf_classifier_1.predict(x_train)


# In[45]:


y_pred_test =rf_classifier_1.predict(x_test)
y_pred_test


# In[46]:


accuracy_score(y_train,y_pred_train)


# In[47]:


confusion_matrix(y_train,y_pred_train)


# In[48]:


print('Classification Report:\n',classification_report(y_train,y_pred_train))


# In[49]:


accuracy_score(y_test,y_pred_test)


# In[50]:


confusion_matrix(y_test,y_pred_test)


# In[51]:


print('Classification Report:\n',classification_report(y_test,y_pred_test))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Problem Statement:
A cloth manufacturing company is interested to know about the segment or
attributes causes high sale. Approach - A Random Forest can be built with target
variable Sales (we will first convert it in categorical variable) & all other
variable will be independent in the analysis.

About the data: Let’s consider a Company dataset with around 10 variables 
    and 400 records. The attributes are as follows:

 Sales -- Unit sales (in thousands) at each location  Competitor Price --
Price charged by competitor at each location  Income -- Community
income level (in thousands of dollars)  Advertising -- Local advertising
budget for company at each location (in thousands of dollars)
 Population -- Population size in region (in thousands)  Price
-- Price company charges for car seats at each site  Shelf
Location at stores -- A factor with levels Bad, Good and Medium 
indicating the quality of the shelving location for the car seats
at each site  Age -- Average age of the local population 
 Education -- Education level at each location 
 Urban -- A factor with levels No and Yes to indicate whether 
the store is in an urban or rural location  
US -- A factor with levels No and Yes to indicate whether
the store is in the US or not


# In[ ]:


Importing Libraries


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Importing Data


# In[2]:


company_data = pd.read_csv('Company_Data.csv')
company_data


# In[3]:


company_data.info()


# In[4]:


company_data.info()


# In[5]:


company_data.describe()


# In[6]:


#pairplot
import seaborn as sns
sns.pairplot(company_data)


# In[8]:


# Correlation analysis for company_data
corr = company_data.corr()
corr


# In[9]:


plt.figure(figsize=(10,6))
sns.heatmap(corr,annot=True)
plt.show()


# In[10]:


#count plot
sns.countplot(company_data['ShelveLoc'])
plt.grid(True)
plt.show()

sns.countplot(company_data['Urban'])
plt.grid(True)
plt.show()

sns.countplot(company_data['US'])
plt.grid(True)
plt.show()


# In[12]:


#Converting Target variable 'Sales' into categories Low, Medium and High.
company_data['Sales'] = pd.cut(x=company_data['Sales'],bins=[0, 6, 12, 17], labels=['Low','Medium', 'High'], right = False)
company_data['Sales']


# In[13]:


sns.countplot(company_data['Sales'])
plt.grid(True)
plt.show()


# In[14]:


company_data['Sales'].value_counts()


# In[15]:


company_data.head()


# In[16]:


#encoding categorical company_data
label_encoder = preprocessing.LabelEncoder()

company_data['Sales'] = label_encoder.fit_transform(company_data['Sales'])
company_data['ShelveLoc'] = label_encoder.fit_transform(company_data['ShelveLoc'])
company_data['Urban'] = label_encoder.fit_transform(company_data['Urban'])
company_data['US'] = label_encoder.fit_transform(company_data['US'])

company_data


# In[18]:


# Input and Output variables
X = company_data.drop('Sales', axis = 1)
y = company_data[['Sales']]


# In[19]:


#splitting the data
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)


# In[20]:


print('x_train_shape:',x_train.shape,'\n y_train_shape:',y_train.shape)


# In[21]:


print('x_test_shape :',x_test.shape,'\n y_test_shape :',y_test.shape)


# In[ ]:


#Building model


# In[22]:


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)


# In[ ]:


#Grid SearchCv


# In[ ]:


To check which criterion is best for our RandomForest Classifier 
and also which Max_depth is best for our RandomForest Classifier.


# In[24]:


from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator = rf_model,
                           param_grid = {'criterion':['entropy','gini'],
                                         'max_depth':[2,3,4,5,6,7,8,9,10]},
                           cv=5)
grid_search.fit(X,y)
print(grid_search.best_params_)
print(grid_search.best_score_)


# In[ ]:


we can take ,criterion: 'gini' & max_depth: 10 to get best fitted model.


# In[ ]:


#New mode


# In[25]:


rf_best_model =RandomForestClassifier(criterion = 'gini',random_state=42,max_depth=10)
rf_best_model.fit(x_train,y_train)


# In[26]:


#prediction train data
pred_train_y = rf_best_model.predict(x_train)


# In[27]:


# Predicticting company data by gini
pred_test_y = rf_best_model.predict(x_test)
pred_test_y


# In[29]:


pd.Series(pred_test_y).value_counts()


# In[30]:


accuracy_score(y_train,pred_train_y)


# In[31]:


confusion_matrix(y_train,pred_train_y)


# In[32]:


print('Classification Report:\n',classification_report(y_train,pred_train_y))


# In[33]:


# Checking accurcy of model
accuracy_score(y_test,pred_test_y)


# In[34]:


confusion_matrix(y_test,pred_test_y)


# In[35]:


print('Classification Report:\n',classification_report(y_test,pred_test_y))


# In[ ]:





# In[ ]:





# In[ ]:





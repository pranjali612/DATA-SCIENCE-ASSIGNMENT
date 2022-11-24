#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
tf.debugging.set_log_device_placement(False) 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Generating reproducible results from same code
tf.random.set_seed(14) 


# In[3]:


raw_data = pd.read_csv("gas_turbines.csv")
raw_data.head() 
#TEY is the variable we should predict.


# In[4]:


df = raw_data.copy() 
df = df.drop(['AFDP','GTEP','TIT','TAT','CDP','CO','NOX'],axis=1)
df.head()


# In[5]:


df.info() #No null values


# In[6]:


df.describe()#we need to normalize the values before modeling


# In[ ]:


This is a Regression Problem. Output TEY is between 175 and 100. There are 15k possible values 
in that range.


# In[7]:


sns.set(rc={'figure.figsize':(20,5)})
sns.boxplot(data=df, orient="v", palette="Set2") 
#No Outliers. But we need to standardize the data


# In[ ]:


#Feature analysis


# In[ ]:


all features in the dataset can be used in model building. This is a 
Regression Problem.


# In[ ]:


#train split dataset


# In[8]:


#A common mistake when configuring a neural network is to first normalize the data before splitting the data.
# Reference 5

X =df.iloc[:,:-1]
Y = df.iloc[:,-1]


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)


# In[9]:


y_train=np.reshape(y_train.to_numpy(), (-1,1)) #https://stackoverflow.com/questions/57192304/numpy-python-exception-data-must-be-1-dimensional
y_test=np.reshape(y_test.to_numpy(), (-1,1)) 


# In[10]:


from sklearn.preprocessing import MinMaxScaler

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

print(scaler_x.fit(X_train))
xtrain_scale=scaler_x.transform(X_train)

print(scaler_x.fit(X_test))
xtest_scale=scaler_x.transform(X_test)

print(scaler_y.fit(y_train))
ytrain_scale=scaler_y.transform(y_train)

print(scaler_y.fit(y_test))
ytest_scale=scaler_y.transform(y_test)


# In[11]:


len(xtrain_scale)


# In[ ]:


#visualizing the data


# In[12]:


sns.pairplot(df,palette='deep')


# In[ ]:


Neural network modelling


# In[13]:


import keras 
from keras.models import Sequential
from keras.layers import Dense
import keras
keras. __version__ #init method is not available in this mdethod


# In[14]:


# create model
model1 = Sequential()
model1.add(Dense(4, input_dim=3, kernel_initializer='normal', activation='relu'))
model1.add(Dense(2106,kernel_initializer='normal', activation='relu'))
model1.add(Dense(1, activation='linear'))
# Compile model
model1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])
# Fit the model
hist1 = model1.fit(xtrain_scale, ytrain_scale, validation_split=0.33, epochs=100, batch_size=150)
#At epoch 50, mse and mae just keeps oscillating back and forth


# In[15]:


model1.summary()


# In[ ]:


#model evaluation


# In[16]:


y_predict = model1.predict(xtest_scale)


# In[17]:


# comparision of prediction and actual values

plt.plot(ytest_scale)
plt.plot(y_predict)
plt.title('Preicted V/s Actual')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()#Neural Network is not good modelfor predicting a regression problem


# In[18]:


print(hist1.history.keys())


# In[19]:


hist1_df = pd.DataFrame(hist1.history)
hist1_df["epoch"]=hist1.epoch
hist1_df.tail()


# In[ ]:


#visualize training history


# In[20]:


# summarize history for Loss

sns.set(rc={'figure.figsize':(6,4)})

plt.plot(hist1.history['loss'])
plt.plot(hist1.history['val_loss'])
plt.title('model1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


# summarize history for loss


# In[21]:


plt.plot(hist1.history['mse'])
plt.plot(hist1.history['val_mse'])
plt.title('model1 mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





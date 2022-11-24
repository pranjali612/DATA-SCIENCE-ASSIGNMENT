#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot


# In[4]:


data=pd.read_csv("ToyotaCorolla.csv",encoding="latin1")
data.head()


# In[5]:


data1=data.iloc[:,[2,3,6,8,12,13,15,16,17]]
data1.dtypes


# In[6]:


data2=data1.rename({"Age_08_04":"age","Quarterly_Tax":"tax","Weight":"weight",'Doors':"door"},axis=1)
data2.head()


# In[7]:


data2.isnull().sum()


# In[8]:


data2.duplicated().value_counts()
data2[data2.duplicated()]


# In[9]:


data2.drop_duplicates(inplace=True)


# In[10]:


data2.corr()


# In[11]:


import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(data2)


# In[12]:


data3 =data2.drop("Gears",axis=1)
data3.head()
data3.corr()


# In[ ]:


#Consider only the above columns and prepare a prediction model 
for predicting Price.


# In[14]:


model=smf.ols("Price~age+KM+HP+cc+door+tax+weight",data=data3).fit()
model.summary()


# In[15]:


m1=smf.ols("Price~cc",data=data3).fit()
m1.summary()


# In[16]:


m2=smf.ols('Price~door',data=data3).fit()
m2.summary()


# In[17]:


data4 = data3.drop("cc",axis=1)
data4.head()


# In[18]:


m3= smf.ols('Price~age+KM+HP+door+tax+weight',data=data4).fit()
m3.summary()


# In[ ]:


#vif calculation


# In[19]:


rsq_age=smf.ols('age~KM+HP+door+tax+weight',data=data4).fit().rsquared
vif_age=1/(1-rsq_age)


# In[20]:


rsq_KM=smf.ols('KM~age+HP+door+tax+weight',data=data4).fit().rsquared
vif_KM=1/(1-rsq_KM)


# In[22]:


rsq_HP=smf.ols('HP~age+KM+door+tax+weight',data=data4).fit().rsquared
vif_HP=1/(1-rsq_HP)


# In[23]:


rsq_door=smf.ols('door~age+KM+HP+tax+weight',data=data4).fit().rsquared
vif_door=1/(1-rsq_door)


# In[24]:


rsq_tax=smf.ols('tax~age+KM+HP+door+weight',data=data4).fit().rsquared
vif_tax=1/(1-rsq_tax)


# In[25]:


rsq_Weight=smf.ols('weight~age+KM+HP+door+tax',data=data4).fit().rsquared
vif_Weight=1/(1-rsq_Weight)


# In[26]:


d1 = {'variables':['age','KM','HP','door','tax','weight'],
     'vif':[vif_age,vif_KM,vif_HP,vif_door,vif_tax,vif_Weight]}


# In[27]:


vif_frame=pd.DataFrame(d1)
vif_frame


# In[ ]:


#residual analysis for normality


# In[28]:


res=m3.resid
res.head(10)


# In[29]:


res.mean()


# In[30]:


import statsmodels.api as sm
qqplot=sm.qqplot(res,line='q')
plt.title('test for normality of residuals(q-qplot)')
plt.show


# In[31]:


sns.boxplot(m3.resid)


# In[32]:


list(np.where(m3.resid<-6000))


# In[ ]:


#residual plot for homoscedasticity


# In[33]:


def get_standardized_values(vals):
    return(vals - vals.mean())/vals.std()


# In[34]:


plt.scatter(get_standardized_values(m3.fittedvalues),
           get_standardized_values(m3.resid))


# In[ ]:


#residual vs regressor


# In[35]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(m3,'age',fig=fig)
plt.show()


# In[36]:


fig = sm.graphics.plot_regress_exog(m3,'KM',fig=fig)
fig


# In[37]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(m3,'weight',fig=fig)
plt.show()


# In[38]:


sm.graphics.plot_regress_exog(m3,'HP',fig=fig)


# In[39]:


sm.graphics.plot_regress_exog(m3,'door',fig=fig)


# In[40]:


sm.graphics.plot_regress_exog(m3,'tax',fig=fig)


# In[41]:


model_influence=m3.get_influence()
(c, _) = model_influence.cooks_distance


# In[42]:


fig=plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(data4)),np.round(c,3))
plt.xlabel("index")
plt.ylabel("cooks distance")


# In[43]:


np.argmax(c),np.max(c)


# In[44]:


data_new=data4.drop(data4.index[220],axis=0)
data_new.head()


# In[45]:


m4 = smf.ols("Price~age+KM+HP+door+tax+weight",data=data_new).fit()
m4.summary()


# In[46]:


influence_plot(m4)


# In[47]:


dataa = data_new.drop(data_new.index[960],axis=0).reset_index()

dataa1 = data_new.drop(data_new.index[958],axis=0).reset_index()


# In[48]:


m5 = smf.ols("Price~age+KM+HP+door+tax+weight",data=dataa1).fit()
m5.summary()


# In[49]:


dataa1['Predicted'] = m5.fittedvalues
dataa1['Errors'] = dataa1["Price"]-dataa1["Predicted"]
dataa1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





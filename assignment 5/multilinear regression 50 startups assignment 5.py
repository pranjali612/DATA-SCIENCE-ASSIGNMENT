#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''Prepare a prediction model for profit of 50_startups data.
Do transformations for getting better predictions of profit and
make a table containing R^2 value for each prepared model.

R&D Spend -- Research and devolop spend in the past few years
Administration -- spend on administration in the past few years
Marketing Spend -- spend on Marketing in the past few years
State -- states from which data is collected
Profit  -- profit of each state in the past few years'''


# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing


# In[4]:


data=pd.read_csv("50_Startups.csv")
data.head(2)


# In[5]:


data.info()


# In[6]:


df = pd.get_dummies(data,["State"])
df.head()


# In[7]:


df.corr()


# In[8]:


df = df.drop("State_New York",axis=1)
df.head()


# In[ ]:


#scatterplots visualizations


# In[9]:


import seaborn as sns 
sns.set_style(style="darkgrid")
sns.pairplot(df)


# In[11]:


df1=df.rename({"R&D Spend":"rd",
                   "Administration":"admin",
                   "Marketing Spend":"ms","State_California":"sc","State_Florida":"sf"},axis=1)


# In[12]:


df1.head(2)


# In[13]:


df1.corr()


# In[14]:


model=smf.ols("Profit~rd+admin+ms+sc+sf",data=df1).fit()
model.summary()


# In[15]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)


# In[16]:


df2=df1.drop(df1.index[49],axis=0)
df2


# In[17]:


m1=smf.ols("Profit~admin+rd+ms+sc+sf",data=df2).fit()
m1.summary()


# In[ ]:


#residual analysis


# In[18]:


res=m1.resid
res


# In[19]:


res.mean() #it is near 0 but not equal to 0


# In[22]:


import statsmodels.api as sm
qqplot=sm.qqplot(res,line="q") 
plt.title("Test for normality of residuals(q-q plot)")
plt.show()


# In[23]:


list(np.where(m1.resid<-15000))


# In[ ]:


#residual plot for homosceasity


# In[26]:


def get_Stardardized_values(vals):
    return(vals - vals.mean())/(vals.std())


# In[27]:


plt.scatter(get_Stardardized_values(m1.fittedvalues),
            get_Stardardized_values(m1.resid))


# In[ ]:


#residual vs Regressor


# In[28]:


fig= plt.figure(figsize=(15,8))
fig=sm.graphics.plot_regress_exog(m1,'admin',fig=fig)
plt.show()


# In[29]:


fig= plt.figure(figsize=(15,8))
fig=sm.graphics.plot_regress_exog(m1,'ms',fig=fig)
plt.show()


# In[30]:


fig=plt.figure(figsize=(15,8))
fig=sm.graphics.plot_regress_exog(m1,'rd',fig=fig)
plt.show()


# In[31]:


sm.graphics.plot_regress_exog(m1,"sc",fig=fig)


# In[32]:


sm.graphics.plot_regress_exog(m1,"sf",fig=fig)


# In[ ]:


#model deletion diagnostic


# In[33]:


from statsmodels.graphics.regressionplots import influence_plot
model_influence =m1.get_influence()
(c,_)=model_influence.cooks_distance


# In[34]:


c


# In[36]:


fig=plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(df2)),np.round(c,3))
plt.xlabel("index")
plt.ylabel("cooks distance")


# In[37]:


np.argmax(c),np.max(c)


# In[38]:


new_data=df2.drop(df2.index[[48]],axis=0).reset_index()


# In[39]:


new_data.head()


# In[41]:


data2=new_data.copy()
data2.head()


# In[42]:


final_m=smf.ols("Profit~rd+admin+ms+sc+sf",data=data2).fit()
final_m.summary()


# In[43]:


final_m.rsquared,final_m.aic


# In[44]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(final_m)


# In[ ]:


#predict for new data


# In[45]:


new_dataa=pd.DataFrame({'rd':162597.7,'admin':151377.59,'ms':443898.53,"sc":1,"sf":0},index=[1])


# In[46]:


new_dataa


# In[47]:


pred_y=final_m.predict(new_dataa)


# In[48]:


pred_y


# In[50]:


data2["predicted"] = final_m.fittedvalues
data2["erro"]=data2["Profit"]-data2["predicted"]
data2.head()


# In[ ]:





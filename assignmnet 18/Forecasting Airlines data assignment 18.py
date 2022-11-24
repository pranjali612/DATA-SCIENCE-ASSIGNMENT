#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from pylab import rcParams
from math import sqrt
from sklearn.metrics import mean_squared_error


# In[2]:


#loading the data
airlines_data = pd.read_excel("Airlines+Data.xlsx")
airlines_data


# In[3]:


airlines_data['Passengers'].plot(figsize=(16,7));


# In[ ]:


The data has a trend and is not stationary.


# In[4]:


airlines_data = airlines_data.set_index('Month')
# SMA over a period of 2 and 3 years 
airlines_data['SMA_2'] = airlines_data['Passengers'].rolling(2, min_periods=1).mean()
airlines_data['SMA_4'] = airlines_data['Passengers'].rolling(4, min_periods=1).mean()
airlines_data['SMA_6'] = airlines_data['Passengers'].rolling(6, min_periods=1).mean()


# In[5]:


#Plotting Simple Moving Averge
colors = ['green', 'red', 'orange','blue']
# Line plot 
airlines_data.plot(color=colors, linewidth=3, figsize=(12,6))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels =['No. of Passengers', '2-years SMA', '4-years SMA','6-years SMA'], fontsize=14)
plt.title('The yearly Passengers travelling', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Passengers', fontsize=16);


# In[6]:


#plotting Cumulative Moving Average
airlines_data['CMA'] = airlines_data['Passengers'].expanding().mean()
airlines_data[['Passengers', 'CMA']].plot( linewidth=3, figsize=(12,6))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels =['Passengers', 'CMA'], fontsize=14)
plt.title('The yearly Passengers travelling', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Population', fontsize=16);


# In[7]:


#Calculating and plotting Exponential MOving average
airlines_data['Ema_0.1'] = airlines_data['Passengers'].ewm(alpha=0.1,adjust=False).mean()
airlines_data['Ema_0.3'] = airlines_data['Passengers'].ewm(alpha=0.3,adjust=False).mean()


# In[8]:


colors = ['#B4EEB4', '#00BFFF', '#FF3030']
airlines_data[['Passengers', 'Ema_0.1', 'Ema_0.3']].plot(color=colors, linewidth=3, figsize=(12,6), alpha=0.8)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels=['No. of passengers', 'EMA - alpha=0.1', 'EMA - alpha=0.3'], fontsize=14)
plt.title('The yearly Passengers travelling.', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Passengers', fontsize=16)


# In[9]:


plot_acf(airlines_data['Passengers'])
plt.show()
plot_acf(airlines_data['Passengers'],lags=30)
plt.show()


# In[10]:


from statsmodels.tsa.seasonal import seasonal_decompose
ts_mul = seasonal_decompose(airlines_data.Passengers,model="multiplicative")
fig = ts_mul.plot()
plt.show();


# In[ ]:


Building Arima Model


# In[11]:


X = airlines_data['Passengers']
size = int(len(X)*0.75)
size


# In[12]:


train , test = X.iloc[0:size],X.iloc[size:len(X)]
#train


# In[16]:


get_ipython().system(' pip install statsmodels')


# In[20]:


from statsmodels.tsa.arima_model import ARIMA


# In[27]:


history = [x for x in train]
history[-1]


# In[28]:


# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    yhat = history[-1]
    predictions.append(yhat)
# observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)


# In[ ]:


Building and comparing multiple models


# In[29]:


final_df = pd.read_excel("Airlines+Data.xlsx")
final_df['Date'] = pd.to_datetime(final_df.Month,format="%b-%y")
final_df['month'] = final_df.Date.dt.strftime("%b") #month extraction
final_df['year'] = final_df.Date.dt.strftime("%y")


# In[30]:


#Boxplot
plt.figure(figsize=(10,7))
plt.subplot(211)
sns.boxplot(x="month",y="Passengers",data=final_df,palette='nipy_spectral')
plt.subplot(212)
sns.boxplot(x="year",y="Passengers",data=final_df,palette='plasma');


# In[31]:


final_df = pd.get_dummies(final_df, columns = ['month'])
#final_df


# In[32]:


from typing_extensions import final
t= np.arange(1,97)
final_df['t']= t
final_df['t_square']= (t *t)
log_Passengers=np.log(final_df['Passengers'])
final_df['log_Passengers'] =log_Passengers
final_df


# In[33]:


Train, Test = np.split(final_df, [int(.75 *len(final_df))])


# In[34]:


#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear


# In[35]:


#Exponential

Exp = smf.ols('log_Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['log_Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[36]:


#Quadratic 

Quad = smf.ols('Passengers~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad


# In[37]:


#Additive seasonality 

add_sea = smf.ols('Passengers~month_Jan+month_Feb+month_Mar+month_Apr+month_May+month_Jun+month_Jul+month_Aug+month_Sep+month_Oct+month_Nov+month_Dec',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['month_Jan','month_Feb','month_Mar','month_Apr','month_May','month_Jun','month_Jul','month_Aug','month_Sep','month_Oct','month_Nov','month_Dec']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[38]:


#Additive Seasonality Quadratic 

add_sea_Quad = smf.ols('Passengers~t+t_square+month_Jan+month_Feb+month_Mar+month_Apr+month_May+month_Jun+month_Jul+month_Aug+month_Sep+month_Oct+month_Nov+month_Dec',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['t','t_square','month_Jan','month_Feb','month_Mar','month_Apr','month_May','month_Jun','month_Jul','month_Aug','month_Sep','month_Oct','month_Nov','month_Dec']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[39]:


##Multiplicative Seasonality
Mul_sea = smf.ols('log_Passengers~month_Jan+month_Feb+month_Mar+month_Apr+month_May+month_Jun+month_Jul+month_Aug+month_Sep+month_Oct+month_Nov+month_Dec',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test[['month_Jan','month_Feb','month_Mar','month_Apr','month_May','month_Jun','month_Jul','month_Aug','month_Sep','month_Oct','month_Nov','month_Dec']]))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['log_Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[40]:


#Multiplicative Additive Seasonality 

Mul_Add_sea = smf.ols('log_Passengers~t+month_Jan+month_Feb+month_Mar+month_Apr+month_May+month_Jun+month_Jul+month_Aug+month_Sep+month_Oct+month_Nov+month_Dec',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test[['t','month_Jan','month_Feb','month_Mar','month_Apr','month_May','month_Jun','month_Jul','month_Aug','month_Sep','month_Oct','month_Nov','month_Dec']]))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['log_Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 


# In[41]:


#Compare the results 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# In[42]:


#ADDITIVE SEASONALITY HAS THE BEST ACCURACY


# In[43]:


model_final = smf.ols('Passengers~t+t_square+month_Jan+month_Feb+month_Mar+month_Apr+month_May+month_Jun+month_Jul+month_Aug+month_Sep+month_Oct+month_Nov+month_Dec',data=Train).fit()
pred_new  = pd.Series(model_final.predict(Test))
pred_new


# In[44]:


predict_data= pd.DataFrame()
predict_data["forecasted_passengers"] = pd.Series(pred_new)


# In[45]:


visualize = pd.concat([Train,predict_data])
visualize


# In[46]:


visualize[['Passengers','forecasted_passengers']].reset_index(drop=True).plot(figsize=(16,8));


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





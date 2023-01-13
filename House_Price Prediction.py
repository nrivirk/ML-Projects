#!/usr/bin/env python
# coding: utf-8

# # Required Libraries

# In[1]:


import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns 
import statistics


# # Loading Dataset

# In[2]:


H_price = pd.read_csv(r"C:\Users\91639\Downloads\housing.csv")  # Loading Dataset


# # First 5 Rows of Dataset

# In[3]:


H_price.head()


# # Last 5 Rows of Dataset

# In[4]:


H_price.tail() # Last 5 Rows of Dataset


# # 5 Random Samlpes of Dataset

# In[5]:


H_price.sample(5) 


# # Shape of Dataset

# In[6]:


H_price.shape # Shape of Dataset


# # Duplicated Rows in Dataset

# In[7]:


H_price.duplicated().sum() # Duplicated Records in Dataset


# # Checking Types of variables in Dataset

# In[8]:


H_price.info()


# # List of Numerical variables in Dataset

# In[9]:


N_Columns = H_price.select_dtypes(["float64","int64"])


# In[10]:


N_Columns # Numerical Columns


# # Categorical Columns in Dataset

# In[11]:


C_Columns = H_price.select_dtypes(["object"])


# In[12]:


C_Columns # List of Categorical Columns


# # Feature Engineering

# # Discovering Null Values

# In[13]:


H_price.isnull().sum() # Sum of Null values in each Columns


# In[14]:


H_price.isnull().mean()*100 # Percentage of Null Values in Dataset


# #  Handling Null Values

# # If we have null Values below 5 Percentage in dataset then it can be removed 

# In[15]:


H_price.dropna(inplace=True)


# In[16]:


H_price.isnull().sum() # Checking Null Values Again


# # Outlier Detection

# #  Descriptive Analytics

# In[17]:


H_price.describe() # Descriptive Analytics gives us 5 Number Summary and range of variables of Dataset


# # Using Boxplot for Outlier detection

# In[18]:


import seaborn


# In[19]:


plt.figure(figsize=(10,10))
seaborn.set(style="whitegrid")
seaborn.boxplot(data = H_price.drop(["median_house_value"],axis=1) ,orient="h")


# # We have many outliers in total_rooms,and population and total_bedrooms columns

# # Handling Outliers colunm by colunm

# In[20]:


plt.figure(figsize=(10,5))
seaborn.set(style="whitegrid")
sns.boxplot(H_price["total_rooms"] ,orient="h")


# # Using IQR METHOD

# In[21]:


np.percentile(H_price["total_rooms"],[25,75]) # 25 percentile and 75 percentile


# In[22]:


Q1 = 1450. # Q1, 25 percentile


# In[23]:


Q3 = 3143. # Q3, 75 percentile


# In[24]:


IQR = Q3-Q1


# In[25]:


Lower_fence = Q1-1.5*(IQR)


# In[26]:


Higher_fence = Q3 + 1.5*(IQR)


# In[27]:


print(Lower_fence,Higher_fence)


# In[28]:


H_price = H_price[H_price["total_rooms"] < 5682.5] # Values less than Higher_fence


# In[29]:


H_price


# In[30]:


H_price.shape # shape of data after removal of outlier from first column


# In[31]:


np.percentile(H_price["total_bedrooms"],[25,75]) # 25 percentile and 75 percentile


# In[32]:


Q1= 288. # Q1, 25 percentile


# In[33]:


Q3 = 594. # Q3, 75 percentile


# In[34]:


IQR = Q3-Q1


# In[35]:


Lower_fence = Q1-1.5*(IQR)


# In[36]:


Higher_fence = Q3 + 1.5*(IQR)


# In[37]:


print(Lower_fence,Higher_fence)


# In[38]:


H_price = H_price[H_price["total_bedrooms"] < 1053.0] # Values less than Higher_fence


# In[39]:


H_price.shape # New shape of Dataset


# In[40]:


np.percentile(H_price["population"],[25,75]) # 25 percentile and 75 percentile


# In[41]:


Q1 = 754.


# In[42]:


Q3 = 1539


# In[43]:


IQR = Q3-Q1


# In[44]:


Lower_fence = Q1-1.5*(IQR)


# In[45]:


Higher_fence = Q3 + 1.5*(IQR)


# In[46]:


print(Lower_fence,Higher_fence)


# In[47]:


H_price = H_price[H_price["population"] < 2716.5] # Values less than Higher_fence


# In[48]:


H_price.shape # New shape of Dataset


# In[49]:


np.percentile(H_price["households"],[25,75]) # 25 percentile and 75 percentile


# In[50]:


Q1 = 267.


# In[51]:


Q3 = 523.


# In[52]:


IQR = Q3 - Q1


# In[53]:


Lower_fence = Q1-1.5*(IQR)


# In[54]:


Higher_fence = Q3 + 1.5*(IQR)


# In[55]:


print(Lower_fence,Higher_fence)


# In[56]:


H_price = H_price[H_price["households"] < 907.0] # Values less than Higher_fence


# In[57]:


H_price.shape


# # USING BOXPLOT  AGAIN AFTER REMOVAL OF OUTLIERS FROM ALL COLUMNS

# In[58]:


plt.figure(figsize=(10,10))
seaborn.set(style="whitegrid")
seaborn.boxplot(data = H_price.drop(["median_house_value"],axis=1) ,orient="h")


# In[59]:


H_price.head()


# # Encoding Categorical Variables

# In[60]:


H_price["ocean_proximity"].value_counts()


# In[61]:


H_price = pd.get_dummies(H_price,columns=["ocean_proximity"],drop_first=True) # Using-Dummies


# In[62]:


H_price.head() 


# In[63]:


H_price.shape


# # Feature Scaling 

# # Using _Min_max scaler

# In[64]:


X= H_price.drop(["median_house_value"],axis=1)
y = H_price["median_house_value"]


# In[65]:


# separate dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=50)


# In[66]:


from sklearn.preprocessing import MinMaxScaler


# In[67]:


Scaler = MinMaxScaler()


# In[68]:


X_train_scaled = Scaler.fit_transform(X_train)
X_test_scaled = Scaler.transform(X_test)


# In[69]:


X_train_scaled


# # Implementing ANN (DEEP LEARNING)

# In[70]:


import tensorflow


# In[71]:


from tensorflow import keras 
from keras import Sequential 
from keras.layers import Dense


# In[72]:


model = Sequential()


# In[73]:


model.add(Dense(12,activation="relu",input_dim=12))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(1,activation="linear"))


# In[74]:


model.summary()


# In[75]:


model.compile(loss="mean_squared_error",optimizer="Adam")


# In[77]:


history1 = model.fit(X_train_scaled,y_train,validation_split=0.2, epochs=500)


# In[78]:


y_pred = model.predict(X_test_scaled)


# In[79]:


from sklearn.metrics import r2_score


# In[80]:


r2_score(y_test,y_pred)


# In[81]:


import tensorflow as tf
import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
import warnings
from mlxtend.plotting import plot_decision_regions
from matplotlib.colors import ListedColormap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
import seaborn as sns


# In[82]:


plt.plot(history1.history['loss'], label='train')
plt.plot(history1.history['val_loss'], label='test')
plt.legend()
plt.show()


# # Early Stopping

# In[83]:


callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00001,
    patience=30,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False
)


# In[84]:


history2 = model.fit(X_train_scaled,y_train,validation_split=0.2, epochs=3000,callbacks=callback)


# In[85]:


y_pred = model.predict(X_test_scaled)


# In[86]:


from sklearn.metrics import r2_score


# In[87]:


r2_score(y_test,y_pred)


# In[88]:


plt.plot(history2.history['loss'], label='train')
plt.plot(history2.history['val_loss'], label='test')
plt.legend()
plt.show()


# # Batch Normalization

# In[92]:


from tensorflow.keras.layers import BatchNormalization


# In[93]:


model = Sequential()


# In[94]:


model.add(Dense(12,activation="relu",input_dim=12))
model.add(BatchNormalization())
model.add(Dense(12,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(12,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1,activation="linear"))
model.summary()


# In[96]:


model.compile(loss="mean_squared_error",optimizer="Adam")


# In[97]:


history3 = model.fit(X_train_scaled,y_train,validation_split=0.2, epochs=,callbacks=callback)


# In[98]:


y_pred = model.predict(X_test_scaled)


# In[99]:


r2_score(y_test,y_pred)


# In[102]:


plt.plot(history3.history['loss'], label='train')
plt.plot(history3.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[103]:





# # Implementing LinearRegression

# In[210]:


from sklearn.linear_model import LinearRegression


# In[211]:


Model = LinearRegression()


# In[212]:


Model.fit(X_train_scaled,y_train)


# In[213]:


y_pred = Model.predict(X_test_scaled)


# In[214]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[215]:


print("R2",r2_score(y_test,y_pred))


# In[521]:


plt.figure(figsize=(8,4))
sns.scatterplot(y_test,y_pred)


# # Implementing XGBOOST REGRESSOR

# In[525]:


get_ipython().system('pip install xgboost')


# In[532]:


import xgboost as xgb


# In[535]:


xg_reg = xgb.XGBRegressor(objective ='reg:squarederror')


# In[536]:


xg_reg.fit(X_train_scaled,y_train)


# In[537]:


y_pred = xg_reg.predict(X_test_scaled)


# In[538]:


print("R2",r2_score(y_test,y_pred))


# In[539]:


plt.figure(figsize=(8,4))
sns.scatterplot(y_test,y_pred)


# # Implementing RandomForest Regressor(Ensemble Technique)

# In[548]:


from sklearn.ensemble import RandomForestRegressor


# In[549]:


R_Regressor = RandomForestRegressor()


# In[550]:


R_Regressor.fit(X_train_scaled,y_train)


# In[551]:


y_pred = R_Regressor.predict(X_test_scaled)


# In[552]:


print("R2",r2_score(y_test,y_pred))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
data = pd.read_csv(r"E:\学习资料\人工智能\大数据竞赛\data_format2\test_format2.csv")
data.head()


# In[5]:


data.info()


# In[6]:


data.isnull().sum(axis=0)


# In[7]:


#首先处理年龄缺省值为空的部分
age_range=data.loc[:,"age_range"].values.reshape(-1,1)
age_range[:5]


# In[8]:


from sklearn.impute import SimpleImputer
imp_0=SimpleImputer(strategy="constant",fill_value=0)
imp_0=imp_0.fit_transform(age_range)
imp_0[:5]


# In[9]:


data.loc[:,"age_range"]=imp_0


# In[10]:


data.isnull().sum(axis=0)


# In[11]:


#处理值为零的年龄缺省
imp_mean = SimpleImputer(missing_values=0) #实例化，默认均值填补
imp_median = SimpleImputer(missing_values=0,strategy="median") #用中位数填补
imp_mode = SimpleImputer(missing_values=0,strategy="most_frequent") #用众数填补


# In[12]:


imp_mean=imp_mean.fit_transform(age_range)
imp_median=imp_median.fit_transform(age_range)
imp_mode=imp_mode.fit_transform(age_range)


# In[13]:


imp_mean[5565299:5565304]


# In[14]:


imp_median[5565299:5565304]


# In[15]:


imp_mode[5565299:5565304]


# In[16]:


#经过以上对比，选择中位数填充年龄
data.loc[:,"age_range"]=imp_median


# In[17]:


data.info()


# In[18]:


data.isnull().sum(axis=0)


# In[19]:


#处理性别缺省部分
gender=data.loc[:,"gender"].values.reshape(-1,1)
gender[:5]


# In[20]:


imp_2=SimpleImputer(strategy="constant",fill_value=2)


# In[21]:


imp_2=imp_2.fit_transform(gender)


# In[49]:


imp_2[6358550:6358555]


# In[23]:


data.loc[:,"gender"]=imp_2


# In[24]:


data.isnull().sum(axis=0)


# In[32]:


#用众数填充性别中的缺省值
imp_mode=SimpleImputer(missing_values=2,strategy="most_frequent")
imp_mode=imp_mode.fit_transform(gender)


# In[50]:


imp_mode[6358550:6358555]


# In[31]:


data.loc[:,"gender"]=imp_mode


# In[51]:


data.isnull().sum(axis=0)


# In[120]:


#label是空的代表的是要预测的用户，不用修改


# In[52]:


#删除activity_log为空的行
data1=data.dropna(subset=["activity_log"])


# In[53]:


data1.isnull().sum(axis=0)


# In[ ]:





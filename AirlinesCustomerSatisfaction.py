#!/usr/bin/env python
# coding: utf-8

# # Airlines Customer Satisfaction

# In[1]:


# importing important packages
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[2]:


#importing the datasets
dataset=pd.read_csv("Invistico_Airline.csv")


# In[3]:


# reading the dataset
dataset


# In[4]:


#shape of dataset
dataset.shape


# In[5]:


#checking the null values in the dataset
dataset.isnull().sum()
#getting the null value in Arrival Delay in minutes column=393


# In[6]:


dataset["Arrival Delay in Minutes"].isnull().sum()


# In[7]:


#replacing those null value from dataset
mean_value=dataset["Arrival Delay in Minutes"].mean()
dataset["Arrival Delay in Minutes"]=dataset["Arrival Delay in Minutes"].fillna(mean_value)


# In[8]:


#checking the null after replacing
dataset.isnull().sum()


# In[9]:


#storing this clean data into new variable
df=dataset
df


# ### Label Encoding

# In[10]:


#Label encoding is a preprocessing step in machine learning where categorical variables 
#(strings or variables that take on a limited set of values) are transformed into numerical values,
#usually integers, in a systematic way. 


# In[11]:


#since we have five columns to transform string value to int value
# 1. Starting from Gender which have Male and Female
gender_transfrom = {"Female": 1, "Male": 0}
df['Gender'] = df['Gender'].map(gender_transfrom) 

# 2. Customer Type
customer_transform={"Loyal Customer":1,"disloyal Customer":0}
df['Customer Type']=df['Customer Type'].map(customer_transform)

# 3. Type of Travel
travel_transform={"Business travel": 2, "Personal Travel": 1}
df["Type of Travel"]=df['Type of Travel'].map(travel_transform)

# 4. class
class_transform = {"Business": 3, "Eco Plus": 2, "Eco": 1}
df['Class'] = df['Class'].map(class_transform)  

# 5. satisfaction
satisfaction_transform={"satisfied":1,"dissatisfied":0}
df['satisfaction']=df['satisfaction'].map(satisfaction_transform)


# In[12]:


#after transforming the values we get:
df


# In[13]:


#spliting the data into features and targets
features=df.drop("satisfaction",axis=1)
targets=df[['satisfaction']]
x_train,x_test,y_train,y_test=train_test_split(features,targets,stratify=targets)


# In[14]:


# featueres
features


# In[15]:


#traget
targets


# ### Logistic Regression

# In[16]:


classifier_model=LogisticRegression(solver='sag',random_state=0)
classifier_model.fit(x_train,y_train)


# In[17]:


#creating prediction model
from sklearn.metrics import accuracy_score
prediction_model=classifier_model.predict(x_test)
accuracy=accuracy_score(y_test,prediction_model)
accuracy


# In[18]:


#creating confusion matrix
confusion_matrix(y_test,prediction_model)


# In[19]:


print(classification_report(y_test,prediction_model))


# In[20]:


# So, it shows 69% accuracy with Precison(74%, 67%) and Recall(50%, 86%) 
# with reference to dissatisified and satisfied respectively.


# In[ ]:





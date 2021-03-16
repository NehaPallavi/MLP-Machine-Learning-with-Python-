#!/usr/bin/env python
# coding: utf-8

# Dataset : Pima Indian Diabetes Dataset from kaggle - https://www.kaggle.com/uciml/pima-indians-diabetes-database

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("datasets_228_482_diabetes.csv",sep=",")


# In[3]:


df.head()


# ## 2

# In[4]:


df.info()


# In[5]:


df.notnull().sum()


# In[6]:


type(df)


# In[7]:


df.shape


# There are no null values in the dataset

# In[8]:


import matplotlib.pyplot as plt


# # Histogram

# In[9]:


plt.hist(df["Glucose"],bins=11)
plt.show()


# In[10]:


plt.hist(df["Pregnancies"],bins=6,color='green')
plt.show()


# In[11]:


plt.hist(df["BloodPressure"],bins=8,color='red')
plt.show()


# In[12]:


plt.hist(df["SkinThickness"],bins=24,color='orange')
plt.show()


# In[13]:


plt.hist(df["Insulin"],bins=12,color='yellow')
plt.show()


# In[14]:


plt.hist(df["BMI"],bins=28,color='violet')
plt.show()


# In[15]:


plt.hist(df["DiabetesPedigreeFunction"],bins=18,color='black')
plt.show()


# In[16]:


plt.hist(df["Age"],bins=20,color='pink')
plt.show()


# In[17]:


df.info()


# # Scatter plot

# In[18]:


plt.scatter(df["Pregnancies"],df["Glucose"])
plt.xlabel("Pregnancies")
plt.ylabel("Glucose")
plt.show()


# In[19]:


plt.scatter(df["BMI"],df["Glucose"],color='green')
plt.xlabel("BMI")
plt.ylabel("Glucose")
plt.show()


# In[20]:


plt.scatter(df["Pregnancies"],df["Age"],color='red')
plt.xlabel("Pregnancies")
plt.ylabel("Age")
plt.show()


# In[21]:


plt.scatter(df["Insulin"],df["Glucose"],color='black')
plt.xlabel("Insulin")
plt.ylabel("Glucose")
plt.show()


# In[22]:


plt.scatter(df["Pregnancies"],df["Insulin"],color='orange')
plt.xlabel("Pregnancies")
plt.ylabel("Insulin")
plt.show()


# # 4

# # 4a)  Splitting the data

# In[23]:


df.head()


# In[24]:


y=df["Outcome"]
x=df.drop("Outcome",axis=1)


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y,random_state=42)


# # Using KNN classification

# In[27]:


from sklearn.neighbors import KNeighborsClassifier


# In[28]:


Knn=KNeighborsClassifier(n_neighbors=5)


# In[29]:


Knn.fit(x_train,y_train)


# In[30]:


Knn.score(x_test,y_test)


# # Using Decision Tree Classifier

# In[31]:


from sklearn.tree import DecisionTreeClassifier


# In[32]:


model=DecisionTreeClassifier()


# In[33]:


model.fit(x_train,y_train)


# In[34]:


model.score(x_test,y_test)


# In[35]:


df.head()


# # Prediction on new dataset

# In[36]:


data={"Pregnancies":[3,2],"Glucose":[95,55],"BloodPressure":[35,96],"SkinThickness":[55,45],"Insulin":[56,3],"BMI":[23.1,45.1],"DiabetesPedigreeFunction":[0.85,0.45],"Age":[52,59]}
data=pd.DataFrame(data)


# In[37]:


data.head()


# In[38]:


Knn.predict(data) #Using Knn classifier


# In[39]:


model.predict(data)  #Using Decision TREE 


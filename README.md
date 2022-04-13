# Cancer-Data-Model
This will have the files for both the analytics model

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


dataset=pd.read_csv("./Cancer_Data.csv")
dataset.head()


# In[5]:


#Shape of Data
dataset.shape 


# In[6]:


#Information
dataset.info


# In[7]:


#Selecting categorical values
dataset.select_dtypes(include='object').columns


# In[8]:


len(dataset.select_dtypes(include='object').columns)


# In[9]:


#Selecting numerical values
dataset.select_dtypes(include=['float64','int64']).columns 


# In[10]:


len(dataset.select_dtypes(include=['float64','int64']).columns)


# In[11]:


dataset.describe()


# In[12]:


#here we can get the list of of all the columns
dataset.columns 


# In[13]:


#here we are checking there is any null value or not
dataset.isnull().values.any() 


# In[14]:


# here we are checking how many null values are in or dataset
dataset.isnull().values.sum() 


# In[15]:


dataset.columns[dataset.isnull().any()]


# In[16]:


#here we are checking how many null columns are in dataset
len(dataset.columns[dataset.isnull().any()]) 


# In[17]:


dataset['Unnamed: 32'].count()


# In[18]:


dataset.drop(columns='Unnamed: 32',inplace=True)


# In[19]:


dataset.shape


# In[29]:


dataset.isnull().values.any()


# In[34]:


#One hot encoding
#It will convert the value in numerical ones
dataset=pd.get_dummies(data=dataset,drop_first=True) 
dataset.head()


# In[35]:


#CountPlot
sns.countplot(dataset['diagnosis_M'],label='count')
plt.show()


# In[36]:


#B(0) vales
(dataset.diagnosis_M==0).sum()


# In[24]:


dataset=pd.get_dummies(data=dataset,drop_first=True) #it will convert the value in numerical ones
dataset.head()


# In[37]:


#m(1) values
(dataset.diagnosis_M==1).sum()


# In[38]:


#Correlation Heatmap and matrix
dataset_2=dataset.drop(columns='diagnosis_M')


# In[39]:


dataset_2.head()


# In[40]:


dataset_2.corrwith(dataset['diagnosis_M']).plot.bar(
    figsize=(20,10),title='correlated with diagnosis_M',rot=45,grid=True
)


# In[41]:


corr=dataset.corr()


# In[42]:


corr


# In[43]:


plt.figure(figsize=(20,10))
sns.heatmap(corr,annot=True)


# In[44]:


#SPlitting the train and test set
dataset.head()


# In[45]:


#Matrix of features or independant variables
x=dataset.iloc[:,1:-1].values


# In[46]:


x.shape


# In[47]:


#target varaiable or dependent variable
y=dataset.iloc[:,-1].values


# In[48]:


y.shape


# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=0
)


# In[51]:


x_train.shape


# In[52]:


y_train.shape


# In[53]:


x_test.shape


# In[54]:


y_test.shape


# In[56]:


#Features Scaling
from sklearn.preprocessing import StandardScaler


# In[57]:


sc=StandardScaler()


# In[58]:


x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[59]:


x_train


# In[60]:


x_test


# In[61]:


#Model Building
#Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[62]:


# here we are succefull created a instance of the class
classifir_lr=LogisticRegression(random_state=0)


# In[63]:


classifir_lr.fit(x_train,y_train)


# In[64]:


y_pred=classifir_lr.predict(x_test)


# In[65]:


from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score


# In[66]:


acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)


# In[67]:


results=pd.DataFrame([['Logistic Regression',acc,f1,prec,rec]],
                    columns=['Model','Accuracy','f1 score','precision','recall'])


# In[68]:


results


# In[69]:


cm= confusion_matrix(y_test,y_pred)
print(cm)


# In[70]:


#Cross Validation
from sklearn.model_selection import cross_val_score


# In[71]:


accuracies= cross_val_score(estimator=classifir_lr,X=x_train,y=y_train,cv=10)


# In[72]:


print("Accuracy is {:.2f}%".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f}%".format(accuracies.std()*100))


# In[73]:


#2. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier


# In[74]:


classifier_rm=RandomForestClassifier(random_state=0)
classifier_rm.fit(x_train,y_train)


# In[75]:


y_pred=classifier_rm.predict(x_test)


# In[76]:


from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score


# In[77]:


acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)


# In[78]:


model_results=pd.DataFrame([['Random forest',acc,f1,prec,rec]],
                    columns=['Model','Accuracy','f1 score','precision','recall'])


# In[79]:


results.append(model_results,ignore_index=True)


# In[80]:


cm= confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


#Cross Validation
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator=classifier_rm,X=x_train,y=y_train,cv=10)
print("Accuracy is {:.2f}%".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f}%".format(accuracies.std()*100))



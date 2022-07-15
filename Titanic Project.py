#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_df=pd.read_csv('train.csv')


# In[3]:


train_df


# In[4]:


train_df.describe()


# In[5]:


sns.heatmap(train_df.corr())


# In[6]:


train_df.isnull()


# In[19]:


sns.heatmap(train_df.isnull(),yticklabels=False,cbar=True)


# In[9]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train_df)


# In[10]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train_df)


# In[11]:


sns.distplot(train_df['Age'],bins=10,kde=False)


# In[12]:


sns.distplot(train_df['Age'].dropna(),bins=40,kde=False)
#removed nan values


# In[13]:


sns.countplot(x='SibSp',data=train_df)


# In[14]:


train_df['Fare'].hist(bins=40)


# In[15]:


#now we will remove null values from age and cabin
#first age coloum filling through boxplot


sns.boxplot(x='Pclass',y='Age',data=train_df)


# In[17]:


test_df=pd.read_csv('test.csv')
test_df


# In[18]:


data=[train_df,test_df]
for dataset in data:
    mean=train_df['Age'].mean()
    std=test_df['Age'].std()
    is_null=dataset['Age'].isnull().sum()
    rand_age=np.random.randint(mean-std,mean+std,size=is_null)
    age_slice=dataset['Age'].copy()
    age_slice[np.isnan(age_slice)]=rand_age
    dataset['Age']=age_slice
    dataset['Age']=train_df['Age'].astype(int)
    train_df['Age'].isnull().sum()


# In[33]:


common='S'
train_df['Embarked']=train_df['Embarked'].fillna(common)
train_df=train_df.drop(['Name','Ticket'],axis=1)
train_df.info()


# In[31]:


test_df=test_df.fillna(test_df['Fare'].mean())
test_df=test_df.drop(['Name','Ticket','Cabin'],axis=1)
test_df.info()


# In[16]:





# In[34]:


#now checking heatmap
sns.heatmap(train_df.isnull(),cmap=None)


# In[37]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train_df['Sex']=le.fit_transform(train_df['Sex'])
train_df['Embarked']=le.fit_transform(train_df['Embarked'])
train_df


# In[38]:


test_df['Sex']=le.fit_transform(test_df['Sex'])
test_df['Embarked']=le.fit_transform(test_df['Embarked'])
test_df


# In[46]:


x_train=train_df.drop(['Survived'],axis=1)
y_train=train_df['Survived']
x_test=test_df.copy()


# In[47]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)


# In[48]:


print(x_train)
print(y_train)


# In[70]:


from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)
predict=classifier.predict(x_test)


# In[71]:


predict


# In[72]:


import numpy as np
import pandas as pd

test_df['Survived']=predict
test_df.head()


# In[73]:


submission =test_df[['PassengerId','Survived']]


# In[74]:


submission


# In[75]:


submission.to_csv('result6.csv',index=False)


#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning the Student Alcohol Consumption Dataset
# - Link to Original Dataset with details of columns: https://www.kaggle.com/uciml/student-alcohol-consumption

# In[315]:


import pandas as pd


# In[316]:


df = pd.read_csv('student-mat.csv')


# In[317]:


df.head()


# In[318]:


df.columns


# ### Removing Unecessary Columns
# - These were columns I felt had little to no correlation to a student's perfeormance

# In[319]:


df = df.drop(['school', 'sex', 'age', 'Mjob', 'Fjob', 'G1','G2', 'reason', 'guardian'], axis=1)


# In[320]:


df.head()


# In[321]:


# identify which columns need to converted to numeric:
# for i in df.columns:
#     if type(df[i].values[0]) == type('str'):
#         print(i)
#         print(df[i].values[0])
#         print('\n')


# ### Converting the Strings within the data to numbers
# - be careful when running this cell because if run more than once, lambda function will be run multiply times
# - for simplicity just run all cells at once

# In[322]:


def binary_convert(x):

    if x == 'yes':
        return 1
    return 0

# 1 if urban, 0 if rural:
df['address'] = df['address'].apply(lambda x: 0 if x=='R' else 1)
# 1 if famsize > 3, 0 if famsize <= 3:
df['famsize'] = df['famsize'].apply(lambda x: 0 if x=='LE3' else 1)
# 1 if parents living together, 0 if not:
df['Pstatus'] = df['Pstatus'].apply(lambda x: 0 if x=='A' else 1)

df['schoolsup'] = df['schoolsup'].apply(binary_convert)

df['famsup'] = df['famsup'].apply(binary_convert)

df['paid'] = df['paid'].apply(binary_convert)

df['activities'] = df['activities'].apply(binary_convert)

df['nursery'] = df['nursery'].apply(binary_convert)

df['higher'] = df['higher'].apply(binary_convert)

df['internet'] = df['internet'].apply(binary_convert)

df['romantic'] = df['romantic'].apply(binary_convert)

# turn data in a binary classification problem, pass/fail
# final grade out of 20
# 1 if grade >= 12, 0 if grade < 12
df['G3'] = df['G3'].apply(lambda x: 0 if x < 12 else 1)


# In[323]:


df.head()

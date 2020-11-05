#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("ACME-HappinessSurvey2020.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.Y.value_counts()


# In[6]:


data.isna().sum()


# In[7]:


data.describe()


# ### Goal is to predict if a customer is happy or not based on the answers they give to questions asked

# In[8]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_score, recall_score,  accuracy_score, precision_recall_curve


# In[9]:


Y = data['Y']
X = data.drop('Y', axis = 1)


# In[10]:


Y.shape, X.shape


# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, random_state = 15)


# In[12]:


print("Train Data Dimensions : ", X_train.shape)
print("Test Data Dimensions : ", X_test.shape)


# In[13]:


logreg = LogisticRegression()
logreg.fit(X_train,Y_train)


# In[14]:


Y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, Y_test)))


# In[15]:


from sklearn.metrics import f1_score
f1_score(Y_test, Y_pred, average='macro')


# In[16]:


import matplotlib.pyplot as plt


# In[17]:


# get importance
importance = logreg.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# ### Gradient boost

# In[19]:


from sklearn.ensemble import GradientBoostingClassifier
gradient_boost = GradientBoostingClassifier(random_state=1)
gradient_boost.fit(X_train, Y_train)
Y_pred = gradient_boost.predict(X_test)
print('Accuracy of gradient boost classifier on test set: {:.2f}'.format(gradient_boost.score(X_test, Y_test)))


# ##### Should the data be divided into 3 sets? training, test and validation set? but dividing into 3 would reduce the data avaliable for training, training set is already small.

# In[29]:


scores = cross_val_score(gradient_boost, X, Y, cv=10)
scores


# In[30]:


print("Accuracy of gradient boosting with 5-fold validation: %0.2f" % (scores.mean()))


# ##### cross fold validation is reducing the accuracy, why? 

# ### Random Forest

# In[32]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state=1)
get_ipython().run_line_magic('time', 'RF.fit(X_train, Y_train)')
Y_pred = RF.predict(X_test)
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(RF.score(X_test, Y_test)))


# In[33]:


scores = cross_val_score(RF, X, Y, cv=10)
scores


# In[43]:


print("Accuracy of Randon Forest with 10-fold validation: %0.2f" % (scores.mean()))


# ##### Again, accuracy reduced with cross validation

# In[36]:


RF.feature_importances_


# In[42]:


print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), RF.feature_importances_), X.columns), 
             reverse=True))


# 

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm


# ### Data exploration

# In[ ]:


# load dataset
df = pd.read_csv("WorldCS2016_ML.csv")


# In[ ]:


df.info()


# In[ ]:


df.head()


# - Do you have missing data?
# - 

# In[ ]:


# Remove those records with missing data
df = df.dropna()


# - How many records do you have now?
# - 

# ### Data Understanding

# In[ ]:


## Correlation Matrix
sns.set(style="white")

# Compute the correlation matrix
corr = df.corr()

sns.heatmap(corr, annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')


# - Which are the three highest correlations (pairs of variables)? 
# - 
# - 
# - 

# In[ ]:


# Correlogram
sns.pairplot(df, diag_kind="kde")


# - Do you observe any outlier in citation? 
# - 

# In[ ]:


df.head()


# Remove the 4 publications with the highest citations.

# In[ ]:


# Drop first row 
# by selecting all rows from first row onwards
df = df.iloc[4: , :]


# In[ ]:


df.head()


# - Which variables in the dataset do not provide any information?
# - 
# 
# Remove not significant variables

# In[ ]:


df = df[['Authors','CiteScore','CiteScorePercentile','Views','OpenAccess','Affiliations','Prominence','Citations','FWCI']]


# ### Simple Linear Regression

# - Select three variables for predicting Citations and buil a simple linear model for each one, except for FWCI.
# - Plot the residuals for the three models

# In[ ]:


# Model 1
independent = 'Views'
feature_cols = [independent]
X = df[feature_cols] # Features (independent variables)
y = df.Citations # Target variable

## Add intercept/constant
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
model.summary()


# In[ ]:


# Regression plots
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, independent, fig=fig)


# In[ ]:


# Model 2


# In[ ]:





# In[ ]:


# Model 3


# In[ ]:





# - Which was the best model? 
# - 
# - Which parameter did you choose to select it?
# - 
# - Did you observed any not-random pattern in the residual plots?
# - 
# - Are the intercepts significant?
# - 

# ## Multiple Linear Regression

# ### Predicting Citations

# Learn a Multiple Regression Model to predict Citations with all features, except FWCI, Views and Prominence.

# In[ ]:


feature_cols = ['Authors','CiteScore','CiteScorePercentile','OpenAccess','Affiliations']

X = df[feature_cols] # Features (independent variables)
y = df.Citations # Target variable

X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
est.summary()


# - Are all coefficients significant?
# -
# - Do you have any multicollinearity problem?
# - 
# 
# Remove non-significant coefficients until getting a valid regression model (all p-values <= 0.05)

# In[ ]:





# In[ ]:





# ### Predicting FWCI
# 
# Learn a Multiple Regression Model to predict FWCI with all features, except Citations, Views and Prominence.

# In[ ]:





# #### Train a valid model without intercept.

# In[ ]:





# In[ ]:





# ## Model selection

# Make a result table with all valid models indicating:
# - Dependent variable (Target)
# - Indepent variable(s)
# - Adj. R2
# - F-Statistics
# - All coefficients significant (p-value >= 0.05)? (Yes/No)
# - Condition No.

# | Target | Predictors | Adj. R2 | F-Statistic | p-value<0.05 | Cond. No. | 
# |---|---|---|---|---|---|
# | x | y |   |   |   |   |
# |---|---|---|---|---|---|

# #### Choose the 3 best models and justify why
# - 
# - 
# - 

# #### Write down the linear equation of these models
# - 
# -
# -

# #### Answer the following questions:
# - Does collaboration increases citation?
# - 
# - Does Open Access improves the citation expectation?
# - 

# In[ ]:





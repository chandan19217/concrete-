#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import seaborn as sns
# ^^^ pyforest auto-imports - don't write above this line
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss


# In[11]:


from pyforest import*
lazy_imports()


# In[12]:


df=pd.read_csv('dataset2.csv')
df.head()


# In[13]:


sns.pairplot(df)
ax = sns.heatmap(df, center=0, cmap="PiYG")
print(ax)

                
 

                  
                      
                     
              
                      
                      


# In[14]:


sns.scatterplot(y="strength", x="cement", hue="water",size="age", data=df, sizes=(50, 300))

                 
                     


# In[17]:


sns.scatterplot(y="strength", x="Fine aggregate", hue="ash",
   size="superplasticizer", data=df,  sizes=(50, 300))
     
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[152]:


sns.boxplot(x='slag',data=df,orient='h')


# #multivariate analysis

# In[153]:


#distplot


# In[154]:


fig,ax2=plt.subplots(3,3,figsize=(16,16))
sns.distplot(df['cement'],ax=ax2[0][0])
sns.distplot(df['slag'],ax=ax2[0][1])
sns.distplot(df['ash'],ax=ax2[0][2])
sns.distplot(df['water'],ax=ax2[1][0])


# In[155]:


df.head()


# In[156]:


#splitting the data into the independant and depandant


# In[157]:


#independant and dependant variables


# In[20]:


X=df.drop('strength',axis=1)
y=df['strength']


# In[23]:


from scipy.stats import zscore
Xscaled=X.apply( zscore)
Xscaled_df=pd.DataFrame(Xscaled,columns=df.columns)
from sklearn.preprocessing import StandardScaler


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(Xscaled,y,test_size=0.3,random_state=1)
 
sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test)


# In[40]:


# Importing models 
from sklearn.linear_model import LinearRegression, Lasso, Ridge 
# Linear Regression 
lr = LinearRegression() 
# Lasso Regression 
lasso = Lasso() 
# Ridge Regression 
ridge = Ridge() 
# Fitting models on Training data 
lr.fit(x_train, y_train) 
lasso.fit(x_train, y_train) 
ridge.fit(x_train, y_train) 
# Making predictions on Test data 
y_pred_lr = lr.predict(x_test) 
y_pred_lasso = lasso.predict(x_test) 
y_pred_ridge = ridge.predict(x_test) 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
print("Model\t\t\t RMSE \t\t R2") 
print("""LinearRegression \t {:.2f} \t\t{:.2f}""".format(  np.sqrt(mean_squared_error(y_test, y_pred_lr)), r2_score(y_test, y_pred_lr))) 
print("""LassoRegression \t {:.2f} \t\t{:.2f}""".format( np.sqrt(mean_squared_error(y_test, y_pred_lasso)), r2_score(y_test, y_pred_lasso))) 
print("""RidgeRegression \t {:.2f} \t\t{:.2f}""".format( np.sqrt(mean_squared_error(y_test, y_pred_ridge)), r2_score(y_test, y_pred_ridge)))
coeff_lr = lr.coef_ 
coeff_lasso = lasso.coef_ 
coeff_ridge = ridge.coef_   

                                                                                                                                                   
                                                                                                                                                   


# ## BUILDING DIFFERENT MODELS

# ## MODEL.FIT(X_TRAIN,Y_TRAIN)

# In[37]:


model=RandomForestRegressor()
model.fit(x_train,y_train)


# In[26]:


y_pred=model.predict(x_test)


# In[27]:


y_pred


# In[165]:


model.score(x_test,y_test)


# In[166]:


from sklearn.metrics import r2_score, mean_squared_error
r2_train = r2_score(y_test, y_pred)
mse_train = mean_squared_error(y_test, y_pred)

mse_train
# In[167]:


print(r2_train)


# In[168]:


print(mse_train)


# In[169]:


#store the accuracy results results for each model in a dataframe in a dataframe for final comparison


# In[170]:


results_1=pd.DataFrame({'Algorithm':['random Forest'],'accuracy':r2_train},index={'1'})
results=results_1[['Algorithm','accuracy']]
results


# KNN REGRESSOR

# In[171]:


from sklearn.neighbors import KNeighborsRegressor
diff_k=[]

for i in range(1,45):
   knn = KNeighborsRegressor(n_neighbors = i)
   knn.fit(x_train,y_train)
   pred_i=knn.predict(x_test)
   diff_k.append(np.mean(pred_i!=y_test))





# In[172]:


diff_k


# In[173]:


plt.figure(figsize=(12,6))
plt.plot(range(1,45),diff_k,color='blue',linestyle='dashed')
plt.title('different k values')


# In[174]:


#k=3 is better choice from the above plot


# In[9]:


model=KNeighborsRegressor(n_neighbors=3)
model.fit(x_train,y_train)


# In[8]:


y_pred=model.predict(x_test)


# In[177]:


y_pred


# In[ ]:





# In[6]:





# In[7]:


y_pred=model.predict(x_test)


# In[181]:


y_pred


# In[182]:


model.score(x_train,y_train)


# In[183]:


model.score(x_test,y_test)


# In[184]:


rq_train = r2_score(y_test, y_pred)
ms_train = mean_squared_error(y_test, y_pred)


# In[185]:


print(rq_train)


# In[186]:


print(ms_train)


# ## XGBOOST REGRESSOR

# In[187]:


import xgboost as xgb
from xgboost.sklearn import XGBRegressor
xgr= XGBRegressor()
xgr.fit(x_train,y_train)


# In[188]:


y_pred=xgr.predict(x_test)


# In[189]:


y_pred


# In[190]:


xgr.score(x_train,y_train)


# In[191]:


xgr.score(x_test,y_test)


# In[192]:


rq_train3 = r2_score(y_test, y_pred)
ms_train3 = mean_squared_error(y_test, y_pred)


# In[193]:


print(rq_train3)


# In[194]:


print(ms_train3)


# ## DECISIONTREE REGRESSION

# In[60]:



from sklearn.tree import DecisionTreeRegressor
dec_model=DecisionTreeRegressor()
dec_model.fit(x_train,y_train)


# In[196]:


#printing the feature importance


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[61]:


print(dec_model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers


# In[62]:


y_pred=dec_model.predict(x_test)


# In[199]:


y_pred


# In[57]:


dec_model.score(x_train,y_train)


# In[63]:


dec_model.score(x_test,y_test)


# In[55]:


df2=df.copy()


# In[56]:


from sklearn.tree import DecisionTreeRegressor
X=df2.drop(['strength','ash','coarse aggregate','Fine aggregate'],axis=1)
Y=df2['strength']
#split x and y into training and test set in 70:30 ratio
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=21)
randomforest=DecisionTreeRegressor()
randomforest.fit(x_train,y_train)
pickle.dump(randomforest,open('titanic_model2.sav','wb'))


# In[50]:


def prediction_model(cement,slag,water,superplasticizer,age):
    import pickle
    x=[[cement,slag,water,superplasticizer,age]]
    randomforest=pickle.load(open('titanic_model2.sav','rb'))
    predictions=randomforest.predict(x)
    print(predictions)


# In[64]:


prediction_model(2,1,1,8,1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





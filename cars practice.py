# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:47:50 2021

@author: DELL
"""

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
cars=pd.read_csv("Cars.csv")
cars.columns
cars.head()

#EDA
cars.isna().sum()
cars.info()
#There are no missing values in the dataset
cars.describe()

#correlation
cars.corr()
type(cars)
#High correlation exists b/w HP and SP, WT and VOL. Hence, there exist high collinearity.

#Visualization(we will use paiplot to visualize multiple features )
import seaborn as sns
sns.pairplot(cars)

#model building(using all the variables)
import statsmodels.formula.api as smf
model= smf.ols("MPG~HP+SP+WT+VOL",data=cars).fit()
model.params
model.summary()
#R2=0.77,adj R2=0.758,p value for WT and VOL is high than 0.05 and also 
#we know that high correlation exist b/w WT and VOL

pred=pd.DataFrame(model.predict(cars))
pred

#attaching the pred column to the dataset
cars["pred"]=pred

#model building (using only VOL)
import statsmodels.formula.api as smf
model_vol=smf.ols("MPG~VOL",data=cars).fit()
model_vol.params
model_vol.summary()
#p is 0.00


#model building(using only WT)
import statsmodels.formula.api as smf
model_WT=smf.ols("MPG~WT",data=cars).fit()
model_WT.params
model_WT.summary()
#p is 0.00
#p is 0.00 when built individually

#model building(using both VOL and WT)
import statsmodels.formula.api as smf
model_VOL_WT=smf.ols("MPG~VOL+WT",data=cars).fit()
model_VOL_WT.params
model_VOL_WT.summary()
#p is greater than 0.05 when built together
#There may be chance of considering only one either VOL or WT.

#We will check for influential data points in the data(using influence index plot)
import statsmodels.api as sm
sm.graphics.influence_plot(model)
#index 76 and 70 are influential points. We will remove those pts from our dataset
cars_new=cars.drop(cars.index[[76,78]],axis=0)
cars_new

#model building again after dropping the influential data pts.
import statsmodels.formula.api as smf
model_new=smf.ols("MPG~HP+SP+WT+VOL",data=cars_new).fit()
model_new.params
model_new.summary()
#r2=0.848, adj r2=0.840
#Still the p values for WT and VOL is greater than 0.05

pred_new=pd.DataFrame(model_new.predict(cars_new))

#Checking the VIF's for independent variables
rsq_hp=smf.ols('HP~SP+WT+VOL',data=cars_new).fit().rsquared
vif_hp=1/(1-rsq_hp)
vif_hp
#16.33

rsq_sp=smf.ols('SP~HP+WT+VOL',data=cars_new).fit().rsquared
vif_sp=1/(1-rsq_sp)
vif_sp
#16.35

rsq_wt=smf.ols('WT~HP+SP+VOL',data=cars_new).fit().rsquared
vif_wt=1/(1-rsq_wt)
vif_wt
#564.98

rsq_vol=smf.ols("VOL~WT+HP+SP",data=cars_new).fit().rsquared
vif_vol=1/(1-rsq_vol)
vif_vol
#564.84
#storing these vif's values in a dataframe
d1={"Variables":["HP","SP","WT","VOL"],"VIF":[vif_hp,vif_sp,vif_wt,vif_vol]}
d1
#we see that vif of WT is higher 

#Added variable plot
sm.graphics.plot_partregress_grid(model_new)
#WT is not showing any significance

#Building a new model by removing the WT 
import statsmodels.formula.api as smf
model_renew=smf.ols("MPG~HP+SP+VOL",data=cars_new).fit()
model_renew.params
model_renew.summary()
#R2=0.848,adj r2=0.842, p is 0.00 
#Also, we see that adj R2 value increased from 0.840 to 0.842

pred_renew=pd.DataFrame(model_renew.predict(cars_new))
pred_renew

#Added varaible plot for the renew model
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model_renew)

#Evaluation of the renew model(rmse)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(cars_new.MPG,pred_renew))
rmse
#3.58
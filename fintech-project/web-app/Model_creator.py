import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# Bagged Trees Regressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler

# Utils
colNames = ["Age","Gender", "Family", "Fin.Edu.", "Risk", "Income", "Wealth"]
colMeans = [33, "Male", 2, 0.41, 0.35, 53, 66]
colMin = [18, "Female", 1, 0, 0, 1, 1]
colMax = [100, "Male", 5, 1, 1, 400, 2200]

#### Baggede model ######
df = pd.read_excel('Needs.xls')
IncomeWealthRatio = np.zeros(df.shape[0])
IncomeWealthRatio[df.Wealth>10] = df.Income[df.Wealth>10]/df.Wealth[df.Wealth>10]
df["IncomeWealth"] = IncomeWealthRatio
scaler = MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df))
X = df.iloc[:,1:8]
X.columns = colNames
X["IncomeWealth"] = df.iloc[:,10]
#Xsmall
Xsmall = X.iloc[:,[0, 3, 5, 6, 7]]
Xsmall = Xsmall[["IncomeWealth", "Age", "Fin.Edu.", "Income", "Wealth"]]
#Train and test
y_inc = df.iloc[:,8] # Income
y_acc = df.iloc[:,9] # Accumulation
X_train, X_test, y_train_inc, y_test_inc = train_test_split(Xsmall, y_inc, random_state=0, test_size=0.3)
X_train, X_test, y_train_acc, y_test_acc = train_test_split(Xsmall, y_acc, random_state=0, test_size=0.3)

# Training for Income
bg_inc = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
bg_inc.fit(X_train,y_train_inc)
print(bg_inc.score(X_test,y_test_inc))      # 0.809
# Training for Accumulation
bg_acc = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
bg_acc.fit(X_train,y_train_acc)
print(bg_acc.score(X_test,y_test_acc))      # 0.843
#bg_inc.predict(X_test)

# Save the model as a pickle in a file
joblib.dump(bg_inc, 'bg_inc.pkl')
joblib.dump(bg_acc, 'bg_acc.pkl')

############ bg #############


##### XGBoost model #####
import xgboost

xgb_acc = xgboost.XGBClassifier().fit(X_train, y_train_acc)
print(xgb_acc.score(X_test, y_test_acc))    # 0.815

xgb_inc = xgboost.XGBClassifier().fit(X_train, y_train_inc)
print(xgb_inc.score(X_test, y_test_inc))    # 0.779


# Save the model as a pickle in a file
joblib.dump(xgb_acc, 'xgb_acc.pkl')
joblib.dump(xgb_inc, 'xgb_inc.pkl')

############### XGB ###############

from sklearn.linear_model import LinearRegression

###### Linear regression #######
df = pd.read_excel('Needs.xls')
scaler = MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df))
id = [1,2,3,4,6,7]
X_Risk = df.iloc[:,id]
X_Risk.columns = ["Age","Gender", "Family", "Financial Education", "Income", "Wealth"]
y_risk = df.iloc[:,5]
model = LinearRegression().fit(X_Risk, y_risk) 
r_sq = model.score(X_Risk, y_risk)
print('coefficient of determination:', r_sq)  # 0.5130044646191534
print('intercept:', model.intercept_)
print('slope:', model.coef_) 

# Save the model
joblib.dump(model, 'lm_risk.pkl')

############# lm ###################


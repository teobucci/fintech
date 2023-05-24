# data management
import pandas as pd
import numpy as np
import joblib
from os.path import join

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler

# Logitboost
from logitboost import LogitBoost

# utils
colNames = ["Age","Gender", "Family", "Fin.Edu.", "Risk", "Income", "Wealth"]
colMeans = [33, "Male", 2, 0.41, 0.35, 53, 66]
colMin = [18, "Female", 1, 0, 0, 1, 1]
colMax = [100, "Male", 5, 1, 1, 400, 2200]





######################################################
#### ------------- Bagged Tree model ------------ ####
######################################################

df = pd.read_excel(join('..','data','Needs.xls'))
df["Income"] = np.power(df.Income, (0.3026))/ 0.3026
df["Wealth"] = np.power(df.Wealth,(0.1341))/ 0.1341
IncomeWealthRatio = np.zeros(df.shape[0])
IncomeWealthRatio[df.Wealth>10] = df.Income[df.Wealth>10]/df.Wealth[df.Wealth>10]
df["IncomeWealth"] = IncomeWealthRatio
scaler = MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df))
X = df.iloc[:,1:8]
X.columns = colNames
X["IncomeWealth"] = df.iloc[:,10]

# Xsmall
Xsmall = X.iloc[:,[0, 3, 5, 6, 7]]
Xsmall = Xsmall[["IncomeWealth", "Age", "Fin.Edu.", "Income", "Wealth"]]

# train test split
y_inc = df.iloc[:,8] # Income
y_acc = df.iloc[:,9] # Accumulation
X_train, X_test, y_train_inc, y_test_inc = train_test_split(Xsmall, y_inc, random_state=0, test_size=0.3)
X_train, X_test, y_train_acc, y_test_acc = train_test_split(Xsmall, y_acc, random_state=0, test_size=0.3)

# training for Income
lboost_inc = LogitBoost(learning_rate=0.019744, n_estimators=200, random_state=0)
lboost_inc.fit(X_train, y_train_inc)
print(lboost_inc.score(X_test,y_test_inc))      # 0.771

# training for Accumulation
bg_acc = BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=36), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
bg_acc.fit(X_train,y_train_acc)
print(bg_acc.score(X_test,y_test_acc))      # 0.793

# save the model as a pickle in a file
joblib.dump(lboost_inc, 'lboost_inc.pkl')
joblib.dump(bg_acc, 'bg_acc.pkl')










######################################################
#### --------------- XGBoost model -------------- ####
######################################################

# We chose to use this models because of its similar performance 
# and better suitability with SHAP library

import xgboost

xgb_acc = xgboost.XGBClassifier().fit(X_train, y_train_acc)
print(xgb_acc.score(X_test, y_test_acc))    # 0.795

xgb_inc = xgboost.XGBClassifier().fit(X_train, y_train_inc)
print(xgb_inc.score(X_test, y_test_inc))    # 0.771

# save the model as a pickle in a file
joblib.dump(xgb_acc, 'xgb_acc.pkl')
joblib.dump(xgb_inc, 'xgb_inc.pkl')










######################################################
#### ---------- Linear Regression model --------- ####
######################################################

from sklearn.linear_model import LinearRegression

df = pd.read_excel(join('..','data','Needs.xls'))
df["Income"] = np.power(df.Income, (0.3026))/ 0.3026
df["Wealth"] = np.power(df.Wealth,(0.1341))/ 0.1341
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

# save the model as a pickle in a file
joblib.dump(model, 'lm_risk.pkl')

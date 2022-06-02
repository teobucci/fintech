# %%
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Bagging
from sklearn.model_selection import train_test_split

# Bagged Trees Regressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.preprocessing import MinMaxScaler

#%%
colNames = ["Age","Gender", "Family", "Fin.Edu.", "Risk", "Income", "Wealth"]
colMeans = [33, "Male", 2, 0.41, 0.35, 53, 66]
colMin = [18, "Female", 1, 0, 0, 1, 1]
colMax = [100, "Male", 5, 1, 1, 400, 2200]

# %%
######### versione 1 Bg ######
df = pd.read_excel('Needs.xls')
df.head()
X = df.iloc[:,1:8]
y = df.iloc[:,9]
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
reg = BaggingRegressor(n_estimators=100, random_state = 0)
reg.fit(X_train, y_train)
y_test_pred = reg.predict(X_test)
score = reg.score(X_test, y_test)
print(score)
estimator_range = [1] + list(range(10, 150, 20))
scores = []
sample_range = list([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

for sample in list(range(10)):
    reg = BaggingClassifier(DecisionTreeClassifier(), n_estimators=60,  max_samples= (sample+1)/10, max_features = 1.0)
    reg.fit(X_train, y_train)
    scores.append(reg.score(X_test, y_test))
plt.figure(figsize = (10,7))
plt.plot(estimator_range, scores)

plt.xlabel('n_estimators', fontsize =20)
plt.ylabel('Score', fontsize = 20)
plt.tick_params(labelsize = 18)
plt.grid()
bg = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
bg.fit(X_train,y_train)
bg.score(X_test,y_test)
bg.predict(X_test.loc[1,:])
###########################

#%%
class Client():

    def __init__(self):
        self.client = [0]*7
        self.income = 0
        self.acc = 0

# %%
man = Client()

# %%
#### Baggede model ######
df = pd.read_excel('Needs.xls')
scaler = MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df))
X = df.iloc[:,1:8]
y_inc = df.iloc[:,8] # Income
y_acc = df.iloc[:,9] # Accumulation
X_train, X_test, y_train_inc, y_test_inc = train_test_split(X, y_inc, random_state=0, test_size=0.3)
X_train, X_test, y_train_acc, y_test_acc = train_test_split(X, y_acc, random_state=0, test_size=0.3)

bg_inc = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
bg_inc.fit(X_train,y_train_inc)
print(bg_inc.score(X_test,y_test_inc))      # 0.792

bg_acc = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
bg_acc.fit(X_train,y_train_acc)
print(bg_acc.score(X_test,y_test_acc))      # 0.800
#bg_inc.predict(X_test)
############ bg #############

# %%
import joblib
# Save the model as a pickle in a file
joblib.dump(bg_inc, 'bg_inc.pkl')
joblib.dump(bg_acc, 'bg_acc.pkl')

#%%
bg_inc = joblib.load('bg_inc.pkl')
bg_acc = joblib.load('bg_acc.pkl')

# %%
from sklearn.linear_model import LinearRegression
#%%

###### Linear regression #######
df = pd.read_excel('Needs.xls')
scaler = MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df))
id = [1,2,3,4,6,7]
X_Risk = df.iloc[:,id]
y_risk = df.iloc[:,5]
model = LinearRegression().fit(X_Risk, y_risk) 
r_sq = model.score(X_Risk, y_risk)
print('coefficient of determination:', r_sq)  # 0.5130044646191534
print('intercept:', model.intercept_)
print('slope:', model.coef_) 
y_pred = model.predict(X_Risk)
#%%
joblib.dump(model, 'lm_risk.pkl')
################################

# %%
Products = pd.read_excel('Needs.xls', sheet_name="Products")


# %%
import shap
#compute SHAP values

# %%
explainer = shap.Explainer(bg_acc, X_test)
#%%
shap_values = explainer(X_test)
# %%
shap.plots.waterfall(shap_values[3], max_display=14)
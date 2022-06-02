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


#%%
##### Prove con Catboost e SHAP ######
df = pd.read_excel('Needs.xls')
scaler = MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df))
X = df.iloc[:,1:8]
X.columns = colNames
y_inc = df.iloc[:,8] # Income
y_acc = df.iloc[:,9] # Accumulation
X_train, X_test, y_train_inc, y_test_inc = train_test_split(X, y_inc, random_state=0, test_size=0.3)
X_train, X_test, y_train_acc, y_test_acc = train_test_split(X, y_acc, random_state=0, test_size=0.3)
# %%
import catboost
from catboost import *
import shap
shap.initjs()
# %%
model = CatBoostClassifier(iterations=70, learning_rate=0.2, random_seed=123)
model.fit(X_train, y_train_inc, verbose=False, plot=False)
model.score(X_test, y_test_inc) # 0.8033333333333333

# %%
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# visualize the first prediction's explanation
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
# %%
# summarize the effects of all the features
shap.summary_plot(shap_values, X_test)

#%%
explainer = shap.TreeExplainer(bg_inc)
shap_values = explainer.shap_values(X_test)
# %%
# summarize the effects of all the features
shap.summary_plot(shap_values, X_test.loc[0,:])




# %%
import sklearn
#%%
df = pd.read_excel('Needs.xls')
scaler = MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df))
# df = shap.sample(df, 1000)
X = df.iloc[:,1:8]
X.columns = colNames
y_inc = df.iloc[:,8] # Income
y_acc = df.iloc[:,9] # Accumulation
X_train, X_test, y_train_inc, y_test_inc = train_test_split(X, y_inc, random_state=0, test_size=0.3)
X_train, X_test, y_train_acc, y_test_acc = train_test_split(X, y_acc, random_state=0, test_size=0.3)

#%%
# print the JS visualization code to the notebook
shap.initjs()
# train a SVM classifier
svm = sklearn.svm.SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train_inc)

# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(svm.predict_proba, X_train, link="logit")
shap_values = explainer.shap_values(X_test, nsamples=10)

# plot the SHAP values for the Setosa output of the first instance
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], link="logit")
# %%


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



#%%
# Training for Income
bg_inc = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
bg_inc.fit(X_train,y_train_inc)
print(bg_inc.score(X_test,y_test_inc))      # 0.809
# Training for Accumulation
bg_acc = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
bg_acc.fit(X_train,y_train_acc)
print(bg_acc.score(X_test,y_test_acc))      # 0.843


#%%
import xgboost
import shap

#%%
model = xgboost.XGBClassifier().fit(X_train, y_train_acc)
model.score(X_test, y_test_acc)

#%%
# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[12], )

# %%
import matplotlib.pyplot as plt


#%%
fig = plt.figure()
plt.subplot(1, 2, 1)
shap.plots.waterfall(shap_values[12],)
plt.subplot(1, 2, 2)
shap.plots.waterfall(shap_values[1],) 
# %%

st.pyplot(fig)
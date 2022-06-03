import pandas as pd
from pyrsistent import inc
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# Interfaccia web
st.title('Product suggestion by Clients\' Needs')

st.markdown("""
This web app helps you suggest the best product based on the client profile.<br />
It takes into account: **Age, Gender, Family Members, Financial Education, Income and Wealth**.<br />
**Risk** is predicted using a linear model.<br />
**Accumulation** and **Income** need are predicted using a Bagging Classifier.<br />
For more indormation about the models see `Model_creator.py` in this repository.<br />
""")

# Utils:
colNames = ["Age", "Gender", "Family Members", "Financial Education", "Income", "Wealth"]       # "Risk",
colMeans = [33, "Male", 2, 0.41, 53, 66]            # 0.35,
colMin = [18, "Female", 1, 0.00, 1, 1]              # 0.00,
colMax = [100, "Male", 5, 1.00, 400, 2200]          # 1.00,
gender = ["Male", "Female"]

# Client to manage informations
class Client():

    def __init__(self):
        self.client = [0] * 6
        self.Xsmall = []
        self.income = 0
        self.accumulation = 0
        self.suggested = pd.DataFrame
        self.risk = 0

man = Client()

# load models for the project
bg_inc = joblib.load('bg_inc.pkl')      # score = 0.792
bg_acc = joblib.load('bg_acc.pkl')      # score = 0.796
lm_risk = joblib.load('lm_risk.pkl')    # r_sq = 0.513
Products = pd.read_excel('Needs.xls', sheet_name="Products")
xgb_acc = joblib.load('xgb_acc.pkl')    # score = 0.815
xgb_inc = joblib.load('xgb_inc.pkl')    # score = 0.779


# Inizialize variable with means
for i in range(len(colNames)):
    if colNames[i] not in st.session_state:
        st.session_state[colNames[i]] = colMeans[i]


# Functions
def CalcRisk():
    # Estimate risk based on other parameters
    id = [0, 1, 2, 3, 4, 5]
    tmp = man.client.iloc[:, id]
    man.risk = lm_risk.predict(tmp)
    man.client["Risk"] = man.risk
    man.client = man.client[["Age", "Gender", "Family Members", "Financial Education", "Risk", "Income", "Wealth"]]
    return

def refactor_solution():
    # factor product df for presentation
    tmp = man.suggested
    tmp["Type"] = ["Income" if i == True else "Accumulation" for i in tmp.Income == 1]
    tmp = tmp[["IDProduct", "Type", "Risk", "Description"]]
    tmp = tmp.set_index(keys="IDProduct")
    man.suggested = tmp.sort_values(by=['Risk'], ascending=False)

def scale():
    # scale client before predicting
    if int(man.client[1] == "Male"):
        man.client[1] = 1
    else:
        man.client[1] = 0
    for i in range(len(colNames)):
        if (i != 1):
            man.client[i] = (man.client[i] - colMin[i])/(colMax[i]-colMin[i])
    man.client = pd.DataFrame(man.client).T
    man.client.columns = ["Age", "Gender", "Family Members", "Financial Education", "Income", "Wealth"]

def createXsmall():
    incWealth = man.client.Income / man.client.Wealth
    # Ratio, Age, Fin, Inc, Wealth
    man.Xsmall = man.client.iloc[:,[0, 3, 5, 6]]
    man.Xsmall["IncomeWealth"] = incWealth
    man.Xsmall = man.Xsmall[["IncomeWealth", "Age", "Financial Education", "Income", "Wealth"]]
    return


def pred():
    # predict client need for inc or acc product
    scale()     
    CalcRisk()
    createXsmall()
    # predict
    # man.income = int(bg_inc.predict(man.Xsmall))
    # man.accumulation = int(bg_acc.predict(man.Xsmall))
    man.income = int(xgb_acc.predict(man.Xsmall))
    man.accumulation = int(xgb_inc.predict(man.Xsmall))
    # query on the result
    tmp = pd.Series([man.risk]*Products.shape[0])
    man.suggested = Products.query(
        "(Income == @man.income or Accumulation == @man.accumulation) and Risk <= @tmp")
    refactor_solution()
    return

# Sidebar
with st.sidebar.form(key='my_form'):
    st.subheader('Client profile')

    # sliders
    for i in range(len(colNames)):
        if (i == 1):
            man.client[i] = st.selectbox('Gender', gender, key=colNames[i])
        else:
            man.client[i] = st.slider(colNames[i],
                                      min_value=colMin[i],
                                      max_value=colMax[i],
                                      key=colNames[i])

    submit_button = st.form_submit_button(label='Calculate!', on_click=pred())


# Show data
if (man.accumulation and man.income):
    st.write(f"The estimate need are: Income and Accumulation")
elif (man.accumulation):
    st.write(f"The estimate need is: Accumulation")
elif (man.income):
    st.write(f"The estimate need is: Income")
else:
    st.write("The client seem not to need products, ask him more informations")
st.write(f"The estimated risk is: {man.risk[0]:.2f}")
st.subheader(f"Suggested products sorted by risk are:")
st.dataframe(man.suggested)

explainer_acc = shap.Explainer(xgb_acc)
shap_values_acc = explainer_acc(man.Xsmall)

explainer_inc = shap.Explainer(xgb_inc)
shap_values_inc = explainer_inc(man.Xsmall)



st.subheader("Explanation of the models")

st.write("Explanation of Income prediction (with rescaled parameters):")
fig_inc = plt.figure()
shap.plots.waterfall(shap_values_inc[0],) 
st.pyplot(fig_inc)

st.write("Explanation of Accumulation prediction (with rescaled parameters):")
fig_acc = plt.figure()
shap.plots.waterfall(shap_values_acc[0],)
st.pyplot(fig_acc)
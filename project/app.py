import pandas as pd
from pyrsistent import inc
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODELS_PATH = PROJECT_ROOT / 'models'
DATA_PATH = PROJECT_ROOT / 'data'

# web interface
st.title('Estimating Clients\' Needs')

st.markdown("""
Original repository: https://github.com/teobucci/fintech/tree/master/project

This web app helps you suggest the best product based on the client profile.
It takes into account:

- Age
- Gender
- Family Members
- Financial Education
- Income and Wealth

As for the outputs:

- **Risk** is predicted using a linear model.
- **Accumulation** and **Income** need are predicted using a Bagging Classifier.

For more indormation about the models see `model_creator.py` in the `models` folder of this repository. 
""")

# utils
colNames = ["Age", "Gender", "Family Members", "Financial Education", "Income", "Wealth"]       # "Risk",
colMeans = [33, "Male", 2, 0.41, 53, 66]            
colMin = [18, "Female", 1, 0.00, 1, 1]             
colMax = [100, "Male", 5, 1.00, 400, 2200]        
gender = ["Male", "Female"]

# client to manage informations
class Client():
    def __init__(self):
        self.client = [0] * 6
        self.Xsmall = [0] * 5
        self.income = 0
        self.accumulation = 0
        self.suggested = pd.DataFrame
        self.risk = 0

man = Client()

# load models
lm_risk = joblib.load(MODELS_PATH / 'lm_risk.pkl')    # r_sq = 0.513
xgb_acc = joblib.load(MODELS_PATH / 'xgb_acc.pkl')    # score = 0.815
xgb_inc = joblib.load(MODELS_PATH / 'xgb_inc.pkl')    # score = 0.779
Products = pd.read_excel(DATA_PATH / 'Needs.xls', sheet_name="Products")


def scale():
    # min-max scale client before predicting
    if int(man.client[1] == "Male"):
        man.client[1] = 1
    else:
        man.client[1] = 0
    # Box-Cox of variables
    man.client[4] = BoxCox(man.client[4], 0.3026)   # Income
    man.client[5] = BoxCox(man.client[5], 0.1341)   # Wealth
    # rescaling
    for i in range(len(colNames)):
        if (i == 4):
            man.client[i] = (man.client[i] - BoxCox(colMin[i], 0.3026))/(BoxCox(colMax[i], 0.3026)-BoxCox(colMin[i], 0.3026))
        elif (i == 5):
            man.client[i] = (man.client[i] - BoxCox(colMin[i], 0.1341))/(BoxCox(colMax[i], 0.1341)-BoxCox(colMin[i], 0.1341))
        elif (i != 1):
            man.client[i] = (man.client[i] - colMin[i])/(colMax[i]-colMin[i])
    man.client = pd.DataFrame(man.client).T
    man.client.columns = ["Age", "Gender", "Family Members", "Financial Education", "Income", "Wealth"]


def CalcRisk():
    # estimate risk based on other parameters
    id = [0, 1, 2, 3, 4, 5] # Age, Gender, Family Members, Fin, Inc, Wealth
    # sample dataframe for risk prediction
    tmp = man.client.iloc[:, id]
    man.risk = lm_risk.predict(tmp)
    # storing information for xgb prediction
    man.client["Risk"] = man.risk
    man.client = man.client[["Age", "Gender", "Family Members", "Financial Education", "Risk", "Income", "Wealth"]]
    return


def createXsmall():
    # create Xsmall dataframe 
    incWealth = man.client.Income / man.client.Wealth
    # rescale incWealth
    incWealth = incWealth / 1.456799689986767 # from Model creation
    # Ratio, Age, Fin, Inc, Wealth
    man.Xsmall = man.client.iloc[:,[0, 3, 5, 6]]
    man.Xsmall["IncomeWealth"] = incWealth
    man.Xsmall = man.Xsmall[["IncomeWealth", "Age", "Financial Education", "Income", "Wealth"]]
    return


def refactor_solution():
    # factor Products df for presentation
    tmp = man.suggested
    tmp["Type"] = ["Income" if i == True else "Accumulation" for i in tmp.Income == 1]
    tmp = tmp[["IDProduct", "Type", "Risk", "Description"]]
    tmp = tmp.set_index(keys="IDProduct")
    man.suggested = tmp.sort_values(by=['Risk'], ascending=False)


def BoxCox (x, p):
    # transform data via Box-Cox, p known
    return np.power(x, p) / p


def pred():
    # predict client need for inc or acc product
    scale()     
    CalcRisk()
    createXsmall()
    # prediction
    man.income = int(xgb_acc.predict(man.Xsmall))
    man.accumulation = int(xgb_inc.predict(man.Xsmall))
    # query on products from the result
    tmp = pd.Series([man.risk]*Products.shape[0])
    man.suggested = Products.query(
        "(Income == @man.income or Accumulation == @man.accumulation) and Risk <= @tmp")
    refactor_solution()
    return




# inizialize variable with means
for i in range(len(colNames)):
    if colNames[i] not in st.session_state:
        st.session_state[colNames[i]] = colMeans[i]

# sidebar
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

    # submit button
    submit_button = st.form_submit_button(label='Calculate!', on_click=pred())


# main page

# estimated needs
if (man.accumulation and man.income):
    st.write(f"The estimate need are: Income and Accumulation")
elif (man.accumulation):
    st.write(f"The estimate need is: Accumulation")
elif (man.income):
    st.write(f"The estimate need is: Income")
else:
    st.write("The client seem not to need products, ask him more informations")
# estimated risk
st.write(f"The estimated risk is: {man.risk[0]:.2f}")
# suggested products
st.subheader(f"Suggested products sorted by risk are:")
st.dataframe(man.suggested)


# shap explanation
explainer_acc = shap.Explainer(xgb_acc)
shap_values_acc = explainer_acc(man.Xsmall)

explainer_inc = shap.Explainer(xgb_inc)
shap_values_inc = explainer_inc(man.Xsmall)

# plots of explanation
st.subheader("Explanation of the models")

st.write("Explanation of **Income** prediction (with rescaled parameters):")
fig_inc = plt.figure()
shap.plots.waterfall(shap_values_inc[0],) 
st.pyplot(fig_inc)

st.write("Explanation of **Accumulation** prediction (with rescaled parameters):")
fig_acc = plt.figure()
shap.plots.waterfall(shap_values_acc[0],)
st.pyplot(fig_acc)

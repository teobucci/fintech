import pandas as pd
import streamlit as st
import numpy as np
import joblib

# Interfaccia web
st.title('Product suggestion by client Needs')

st.markdown("""
This web app helps you to suggest the best product based on the client profile.
It take in account: Age, Gender, Family members, Financial education, Income and Wealth.
""")

# Utils:
colNames = ["Age", "Gender", "Family Members", "Financial Education", "Risk", "Income", "Wealth"]
colMeans = [33, "Male", 2, 0.41, 0.35, 53, 66]
colMin = [18, "Female", 1, 0.00, 0.00, 1, 1]
colMax = [100, "Male", 5, 1.00, 1.00, 400, 2200]
gender = ["Male", "Female"]

# Client to manage informations
class Client():

    def __init__(self):
        self.client = [0]*7
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

# Inizialize variable with means
for i in range(len(colNames)):
    if colNames[i] not in st.session_state:
        st.session_state[colNames[i]] = colMeans[i]


# Functions
def CalcRisk():
    # Estimate risk based on other parameters
    tmp = pd.DataFrame(man.client).T
    id = [0, 1, 2, 3, 5, 6]
    tmp = tmp.iloc[:, id]
    man.risk = lm_risk.predict(tmp)
    return man.risk

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


def pred():
    # predict client need for inc or acc product
    st.write(f"Il soggeto scelto è: {man.client}")
    scale()
    st.write(f"Riscalato diventa: {man.client}")
    risk_pred = CalcRisk()
    st.write(f"Il rischio calcolato è: {risk_pred}")
    man.income = int(bg_inc.predict([man.client]))
    man.accumulation = int(bg_acc.predict([man.client]))
    st.write(f"Income {man.income} e Acc {man.accumulation}")
    man.suggested = Products.query(
        "(Income == @man.income or Accumulation == @man.accumulation) and Risk <= @man.client[4]")
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
st.write(f"The estimate need are: income: {man.income}, accumulation: {man.accumulation}")
st.write(f"The estimated risk is: {man.risk}")
st.write(f"Suggested products are:")
st.dataframe(man.suggested)


# if st.button('Calcola piano'):
#     st.write("Pulsante premuto")

# streamlit run "/Users/marco/Library/CloudStorage/OneDrive-PolitecnicodiMilano/Polimi/4.2 Fintech - Marazzina/Final work/Project_2_py/main_Streamlit.py"

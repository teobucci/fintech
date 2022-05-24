import pandas as pd
import streamlit as st
import base64
import scripts.compilatore as mylib
from math import floor
from os.path import join

# Interfaccia web
st.markdown('# Study Plan maker')
st.markdown("### for the MSc in Mathematical Engineering")

CFU_max_tot = st.number_input('Max Total CFUs', min_value=120, max_value=122, value=121, step=1)
CFU_max_sem = st.number_input('Max CFUs per semester', min_value=30, max_value=80, value=35, step=1)


@st.cache
def format_func(track):
    return {
        'MCS': 'MCS - Computational Science and Computational Learning',
        'MMF': 'MMF - Quantitative Finance',
        'MST': 'MST - Statistical Learning'
    }[track]


track_choice = st.selectbox('Major', ('MCS', 'MMF', 'MST'), 2, format_func=format_func)

st.write('Download the example file and fill in the ``Rating`` column.')

PATH = join('assets', 'source.csv')
df_base = pd.read_csv(PATH, header=0)
df_base['Rating'] = 0


@st.cache
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="study_plan_source_example.csv">Download Example Source CSV File</a>'
    return href


#st.markdown(filedownload(df_base), unsafe_allow_html=True)

st.download_button('Download Example Source CSV File', data=df_base.to_csv(index=False).encode('utf-8'), file_name='study_plan_source_example.csv',)

st.write("To easily edit the CSV file without messing up the formatting, you can use this tool https://www.convertcsv.com/csv-viewer-editor.htm")

df = df_base

uploaded_file = st.file_uploader('Upload your input CSV file', type=['csv'], accept_multiple_files=False)
if uploaded_file:
    df = pd.read_csv(uploaded_file).dropna()
    st.success('File uploaded correctly.')


st.dataframe(df)

DEEP = st.number_input('How many sub-optimal plans would you like to compute? (default: 0)', min_value=0, max_value=5, value=0, step=1)

if st.button('Compute the best Study Plan!'):
    if uploaded_file:
        plans, objective = mylib.generate_plan(df, track=track_choice, CFU_max_sem=CFU_max_sem, CFU_max_tot=CFU_max_tot, N_SUBOPTIMAL=DEEP)

        if len(plans) == 0:
            st.error('Problem is Infeasible. This usually means that you have either too many constraints (eg. CFUs per month is too low) or that you haven\'t chosen enough courses from both 1st and 2nd semester.')
        elif len(plans) == DEEP + 1:
            st.success('Problem is Optimal! And the desired number of sub-optimal solutions has been computed.')

            for plan, obj in zip(plans, objective):
                st.dataframe(plan)
                cfu_tot = sum(plan['CFU']*plan['%'])
                cfu_sem = plan.groupby(['Anno', 'Sem'])['CFU'].sum()
                st.write(f'Total interest: {obj}')
                st.write(f'The total number of CFUs is: {cfu_tot}')
                st.write(f'Here is the number of CFUs per semester:')
                st.dataframe(cfu_sem)
                st.write(mylib.get_exchangable_exams(plan, df, track_choice))
                #st.download_button('Download the generated Study Plan', data=piano.to_csv(index=False).encode('utf-8'), file_name='study_plan_output.csv',)
                st.markdown("""---""")
        else:
            st.success('Problem is Optimal!')
            st.warning('But the desired number of sub-optimal solutions could not be computed.')
    else:
        st.error('Please upload a valid input and try again.')

with st.expander("See explanation of the model"):
    st.write(
        """
**Sets**

- $I$: set of courses.

- $J$: set of years.

- $K$: set of groups.



**Parameters**

- $\mathrm{rating}_{i}$: rating of course $i$.

- $\mathrm{cfu}_{i}$: number of CFUs of course $i$.

- $\mathrm{semester}_{i}$: semester in which course $i$ is erogated.

- $\mathrm{minCFU}_{k}$: minimum number of CFUs for group $k$.

- $\mathrm{crosstab}_{ik}$: is $1$ if course $i$ belongs to group $k$, $0$ otherwise.



**Variables**

- $x_{ijk} \in [ 0,1]$: percentage of course $i$ in year $j$ in group $k$.

- $y_{i} \in \{0,1\}$: is $1$ if course $i$ is chosen, $0$ otherwise.

- $z_{ij} \in \{0,1\}$: is $1$ if course $i$ is chosen in year $j$, $0$ otherwise.



**Objective function**
$$
\max\sum\limits _{i,j,k} x_{ijk}\mathrm{\cdotp rating}_{i} \cdotp \mathrm{cfu}_{i}
$$


**Constraints**
$$
x_{ijk} =1,\\ \\ \\forall i\in \\{\\text{compulsory courses in year } j\\},\\forall k\in K,\\forall j\in J\\\\
\sum\limits _{i\in I,j\in J,k\in K} x_{ijk}\mathrm{\cdotp cfu}_{i} \leq \\text{CFU\_max\_tot}\\\\
\sum\limits _{i\in I\cap \\{\\text{sem. }1\\},k\in K} x_{ijk}\mathrm{\cdotp cfu}_{i} \leq \\text{CFU\_max\_sem},\\ \\ \\forall j\in J\\\\
\sum\limits _{i\in I\cap \\{\\text{sem. }2\\},k\in K} x_{ijk}\mathrm{\cdotp cfu}_{i} \leq \\text{CFU\_max\_sem},\\ \\ \\forall j\in J\\\\
\sum\limits _{k\in K} x_{ijk} \leq 1000\cdotp z_{ij},\\ \\ \\forall i\in I,\\forall j\in J\\\\
\sum\limits _{j\in J} z_{ij} \leq 1,\\ \\ \\forall i\in I\\\\
\sum\limits _{j\in J,k\in K} x_{ijk} \leq y_{i},\\ \\ \\forall i\in I\\\\
y_{i} \leq \sum\limits _{j\in J,k\in K} x_{ijk},\\ \\ \\forall i\in I\\\\
\sum\limits _{i\in I,j\in J} x_{ijk}\mathrm{\cdotp cfu}_{i} \geq \mathrm{minCFU}_{k},\\ \\ \\forall k\in K\\\\
\sum\limits _{j\in J} x_{ijk} \leq \mathrm{crosstab}_{ik},\\ \\ \\forall i\in I,\\forall k\in K
    
$$
"""
    )

with st.expander("About"):
    st.markdown("This is a student project not affiliated with Politecnico di Milano.")
    link = 'Created by [Teo Bucci](https://github.com/teobucci)'+' '+'[Filippo Cipriani](https://github.com/SmearyTundra)'+' '+'[Marco Lucchini](https://github.com/marcolucchini)'
    st.markdown(link, unsafe_allow_html=True)

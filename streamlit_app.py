#WEB APP FOR ERST EDA and Model

#Import libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

import io

#DEFINING FUNCTIONS
#________________________________________________________________________________________________________________________
def df_info(df):
    df.columns = df.columns.str.replace(' ', '_')
    buffer = io.StringIO() 
    df.info(buf=buffer)
    s = buffer.getvalue() 

    df_info = s.split('\n')

    counts = []
    names = []
    nn_count = []
    dtype = []
    for i in range(5, len(df_info)-3):
        line = df_info[i].split()
        counts.append(line[0])
        names.append(line[1])
        nn_count.append(line[2])
        dtype.append(line[4])

    df_info_dataframe = pd.DataFrame(data = {'#':counts, 'Column':names, 'Non-Null Count':nn_count, 'Data Type':dtype})
    return df_info_dataframe.drop('#', axis = 1)

def df_isnull(df):
    res = pd.DataFrame(df.isnull().sum()).reset_index()
    res['Percentage'] = round(res[0] / df.shape[0] * 100, 2)
    res['Percentage'] = res['Percentage'].astype(str) + '%'
    return res.rename(columns = {'index':'Column', 0:'Number of null values'})

def number_of_outliers(df):
    
    df = df.select_dtypes(exclude = 'object')
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    ans = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    df = pd.DataFrame(ans).reset_index().rename(columns = {'index':'column', 0:'count_of_outliers'})
    return df

def space(num_lines=1):
    for _ in range(num_lines):
        st.write("")

def sidebar_space(num_lines=1):
    for _ in range(num_lines):
        st.sidebar.write("")


def sidebar_multiselect_container(massage, arr, key):
    
    container = st.sidebar.container()
    select_all_button = st.sidebar.checkbox("Select all for " + key + " plots")
    if select_all_button:
        selected_num_cols = container.multiselect(massage, arr, default = list(arr))
    else:
        selected_num_cols = container.multiselect(massage, arr, default = arr[0])

    return selected_num_cols 

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Angiv kriterier")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filtrer baseret på", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Værdier for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Værdier for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Værdier for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Indgår i {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

#_________________________________________________________________________________

# stopwords
stopwords = ["ad",
"af","aldrig","alene","alle","allerede","alligevel","alt","altid","anden","andet",
"andre","at","bag","bare","begge","bl.a.","blandt","blev","blive","bliver","burde","bør","ca.","da","de","dem","den","denne","dens","der",
"derefter","deres","derfor","derfra","deri","dermed","derpå","derved","det","dette","dig","din","dine","disse","dit","dog","du","efter","egen","ej","eller","ellers",
"en","end","endnu","ene","eneste","enhver","ens","enten","er","et","f.eks.","far","fem","fik","fire","flere","flest","fleste","for","foran","fordi","forrige",
"fra","fx","få","får","før","først","gennem","gjorde","gjort","god","godt","gør","gøre","gørende","ham","han","hans","har","havde","have","hej","hel",
"heller","helt","hen","hende","hendes","henover","her","herefter","heri","hermed","herpå","hos","hun","hvad","hvem","hver","hvilke","hvilken","hvilkes","hvis","hvor","hvordan",
"hvorefter","hvorfor","hvorfra","hvorhen","hvori","hvorimod","hvornår","hvorved","i","igen","igennem","ikke","imellem","imens","imod","ind","indtil","ingen",
"intet","ja","jeg","jer","jeres","jo","kan","kom","komme","kommer","kun","kunne","lad","langs","lav","lave","lavet","lidt","lige","ligesom","lille",
"længere","man","mand","mange","med","meget","mellem","men","mens","mere","mest","mig","min","mindre","mindst","mine","mit","mod","må","måske","ned",
"nej","nemlig","ni","nogen","nogensinde","noget","nogle","nok","nu","ny","nyt","når","nær","næste","næsten","og","også","okay","om","omkring","op",
"os","otte","over","overalt","pga.","på","RT","samme","sammen","se","seks","selv","selvom","senere","ser","ses","siden","sig","sige","sin","sine",
"sit","skal","skulle","som","stadig","stor","store","synes","syntes","syv","så","sådan","således","tag","tage","temmelig","thi","ti","tidligere","til",
"tilbage","tit","to","tre","ud","uden","udover","under","undtagen","var","ved","vi","via","vil","ville","vor","vore""vores","vær","være","været","øvrigt"]


#_________________________________________________________________________________

#Header
st.write("""
# ERST - SMV:Digital dataoverblik 
""")

uploaded_file = st.file_uploader("Vælg fil")
if uploaded_file is not None:
    init_df = pd.read_csv(uploaded_file, encoding = "UTF-8")


df = filter_dataframe(init_df)

st.dataframe(df)

rowcount = len(df)
st.write(f"### {rowcount} virksomheder opfylder disse kriterier")

st.write("Hvad kendetegner disse virksomheder?")


st.write("### Barrierebeskrivelser")

st.write()

Barr = df.iloc[:,34]

Barr = Barr.dropna()

rowcount_barr = len(Barr)
st.write(f"{rowcount_barr} ud af {rowcount} virksomheder har udfyldt dette felt")
st.dataframe(Barr)

#Count word frequency in Barr

#removing stopwords:
Word_freq_bar = Barr.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

Word_freq_bar = Word_freq_bar.str.split(expand=True).stack().value_counts()

Word_freq_bar = Word_freq_bar.astype(str)



st.write("### Word frequency i Barrierebeskrivelser")
st.write(Word_freq_bar)




#st.dataframe(df)

st.sidebar.header('Visualisering af forløb')

if uploaded_file is not None:
    
    df = df
   
        
    
    st.subheader('Alle variabler for udvalgte virksomheder:')
    n, m = df.shape
    st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)   
    st.dataframe(df)


    all_vizuals = ['NA Info', 'Descriptive Analysis', 'Target Analysis', 
                   'Distribution of Numerical Columns', 'Count Plots of Categorical Columns', 
                   'Box Plots', 'Visualize Correlation']
    sidebar_space(3)         
    vizuals = st.sidebar.multiselect("Vælg visualisering 👇", all_vizuals)

    #if 'Info' in vizuals:
    #    st.subheader('Info:')
    #    c1, c2, c3 = st.columns([1, 2, 1])
    #    c2.dataframe(df_info(df))

    if 'NA Info' in vizuals:
        st.subheader('NA Value Information:')
        if df.isnull().sum().sum() == 0:
            st.write('There is not any NA value in your dataset.')
        else:
            c1, c2, c3 = st.columns([0.5, 2, 0.5])
            c2.dataframe(df_isnull(df), width=1500)
            space(2)
            

    if 'Descriptive Analysis' in vizuals:
        st.subheader('Descriptive Analysis:')
        st.dataframe(df.describe())
        
    if 'Target Analysis' in vizuals:
        st.subheader("Select target column:")    
        target_column = st.selectbox("", df.columns, index = len(df.columns) - 1)
    
        st.subheader("Histogram of target column")
        fig = px.histogram(df, x = target_column)
        c1, c2, c3 = st.columns([0.5, 2, 0.5])
        c2.plotly_chart(fig)


    num_columns = df.select_dtypes(exclude = 'object').columns
    cat_columns = df.select_dtypes(include = 'object').columns

    if 'Distribution of Numerical Columns' in vizuals:

        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = sidebar_multiselect_container('Choose columns for Distribution plots:', num_columns, 'Distribution')
            st.subheader('Distribution of numerical columns')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_num_cols)):
                        break

                    fig = px.histogram(df, x = selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width = True)
                    i += 1

    if 'Count Plots of Categorical Columns' in vizuals:

        if len(cat_columns) == 0:
            st.write('There is no categorical columns in the data.')
        else:
            selected_cat_cols = sidebar_multiselect_container('Choose columns for Count plots:', cat_columns, 'Count')
            st.subheader('Count plots of categorical columns')
            i = 0
            while (i < len(selected_cat_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_cat_cols)):
                        break

                    fig = px.histogram(df, x = selected_cat_cols[i], color_discrete_sequence=['indianred'])
                    j.plotly_chart(fig)
                    i += 1

    if 'Box Plots' in vizuals:
        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = sidebar_multiselect_container('Choose columns for Box plots:', num_columns, 'Box')
            st.subheader('Box plots')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:
                    
                    if (i >= len(selected_num_cols)):
                        break
                    
                    fig = px.box(df, y = selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width = True)
                    i += 1

    #if 'Outlier Analysis' in vizuals:
    #    st.subheader('Outlier Analysis')
    #    c1, c2, c3 = st.columns([1, 2, 1])
    #    c2.dataframe(number_of_outliers(df))

    if 'Visualize Correlation' in vizuals:
        
        
        df_1 = df#.dropna()
        
        
        #high_cardi_columns = []
        #normal_cardi_columns = []

        #for i in cat_columns:
        #    if (df[i].nunique() > df.shape[0] / 10):
        #        high_cardi_columns.append(i)
        #    else:
        #        normal_cardi_columns.append(i)


        #if len(normal_cardi_columns) == 0:
        #    st.write('There is no categorical columns with normal cardinality in the data.')
        #else:
        
        st.subheader('Visualize Correlation of target columns')
            #model_type = st.selectbox('Vælg type:', ("Regression"), key = 'model_type')
        model_type = 'Regression'
        selected_cat_cols = sidebar_multiselect_container('Choose columns for Category Colored plots:', df_1.columns, 'Category')
            
        if 'Target Analysis' not in vizuals: 
        	target_column_x = st.selectbox("Vælg X:", df.columns, index = len(df.columns) - 1)
        	target_column_y = st.selectbox("Vælg Y:", df.columns, index = len(df.columns) - 1)
            
        
        #i = 0
        #while (i < len(selected_cat_cols)):
            
            
        
            #if model_type == 'Regression':
        fig = px.scatter(df_1, x= target_column_x, y = target_column_y, trendline="ols")
        #    else:
        #        fig = px.histogram(df_1, color = selected_cat_cols[i], x = target_column)

        st.plotly_chart(fig, use_container_width = True)
            #    i += 1

            #if high_cardi_columns:
            #    if len(high_cardi_columns) == 1:
            #        st.subheader('The following column has high cardinality, that is why its boxplot was not plotted:')
            #    else:
            #        st.subheader('The following columns have high cardinality, that is why its boxplot was not plotted:')
            #    for i in high_cardi_columns:
            #        st.write(i)
                
            #    st.write('<p style="font-size:140%">Do you want to plot anyway?</p>', unsafe_allow_html=True)    
            #    answer = st.selectbox("", ('No', 'Yes'))

            #    if answer == 'Yes':
            #        for i in high_cardi_columns:
            #            fig = px.box(df_1, y = target_column, color = i)
            #            st.plotly_chart(fig, use_container_width = True)
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import random
import string

import pickle
from st_aggrid import AgGrid, GridUpdateMode, GridOptionsBuilder, DataReturnMode, JsCode

from st_aggrid.grid_options_builder import GridOptionsBuilder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

model = pickle.load(open('RandomForest model.pkl', 'rb'))


#@st.cache
#st.title('Welcome to the modeling page')
def show_model():

    #st.subheader("Modelling")
    st.header('Welcome to the Modeling page')
    if st.sidebar.checkbox('Upload CSV or Excel file'):
        st.subheader('Data uploading')
        with st.sidebar.header('Upload your CSV data for this app'):
            uploaded_file = st.sidebar.file_uploader("Upload your input CSV or XLSX file", type=["csv", "xlsx"])
            st.sidebar.markdown("""

            """)
        st.write('---')

        global df
        global df_new
        #global pre

        if uploaded_file is not None:

            try:
                def load_csv():
                    csv = pd.read_csv(uploaded_file)
                    return csv
                df = load_csv()
            except Exception as e: # -*- coding: utf-8 -*-
                print (e)
                def load_xlsx():
                    xlsx = pd.read_excel(uploaded_file)
                    return xlsx
                df = load_xlsx()

            #temp_df = df.copy
        if st.checkbox("Show dataset"):
            if uploaded_file is None:
                st.warning('Please Upload your Data')
            else:
                st.warning('Uncheck to edit Customer data')
                my_grid =  GridOptionsBuilder.from_dataframe(df)
                AgGrid(df)
                if st.button('Predict', key =1):
                    id_df = pd.DataFrame(df)
                    mydf=pd.read_csv("../project_app_df.csv")
                    mydf=mydf.append(id_df, ignore_index = True)
                    mydf.to_csv("../project_app_df.csv", index = False)
                    model = pickle.load(open('RandomForest model.pkl', 'rb'))
                    out_df=id_df.copy()
                    out_df= out_df.drop(columns=['Customer ID','Gender', 'Age',  'Number of Dependents','Referred a Friend', 'Total Revenue', 'Tenure in Months'])
                    def final():
                        encode_features = [feature for feature  in out_df.columns if out_df[feature ].dtype=='O' and out_df[feature].nunique() == 2]
                        dumy_features = [feature for feature  in out_df.columns if out_df[feature ].dtype=='O' and out_df[feature].nunique() >2]
                        le = LabelEncoder()
                        out_df[encode_features]=out_df[encode_features].apply(le.fit_transform)
                        df_data = pd.get_dummies(out_df, columns=dumy_features, dummy_na=False)
                        final_df= df_data
                        return final_df
                    out_df = final()
                    i = (len(id_df))

                    pre_df = out_df.tail(i)
                    pre_df = pre_df.fillna(0)
                    scaler=MinMaxScaler()
                    pre_df=scaler.fit_transform(pre_df)
                    out_pred= model.predict(pre_df)
                    out_prob= model.predict_proba(pre_df)
                    id_df['Predicted'] = out_pred
                    id_df['Churn_Probability'] = (out_prob[:,1]).round(2)


                    def label(row):
                        if row['Predicted'] == 1:
                            result= 'Yes'
                        else:
                            result= 'No'
                        return result
                    id_df['Churn Label'] = id_df.apply(label, axis = 1)
                    id_df['Count'] = 1
                    pre= pd.DataFrame(id_df)
                    #st.write(pre)
                    new_df = pd.read_csv("../project_report_df.csv")
                    new_df=new_df.append(pre, ignore_index=True)
                    new_df.to_csv("../project_report_df.csv", index = False)
                    st.subheader("Prediction on Customer(s) Data" )
                    if i > 1:
                        result = id_df[['Customer ID', 'Churn_Probability', 'Churn Label']]
                        AgGrid(result)
                    else:
                        if out_pred == 0:
                            st.write('This Customer will not Churn with the probability of')
                            st.success(out_prob[:,1])
                        else:
                            st.write('**This Customer will Churn with the probability of**')
                            st.warning(out_prob[:,1])

                    st.write('---')
                    pre= pd.DataFrame(id_df)
                    st.write('Data to be written',pre)
                    new_df = pd.read_csv("../project_report_df.csv")
                    x= list(new_df['Customer ID'])
                    y = pre['Customer ID'].tolist()
                    res = [ele for ele in x if(ele in y)]
                    if (bool(res)) == True:
                        st.warning('Customer ID(s) below already exist')
                        st.warning('**Warning: The new data below will overwrite existing Customer data**')
                        sub=new_df[new_df['Customer ID'].isin (res)]
                        st.write(sub)
                        new_df= new_df.drop(index= new_df[new_df['Customer ID'].isin(res)].index)

                    new_df=new_df.append(pre, ignore_index=True)
                    new_df.to_csv("../project_report_df.csv", index = False)

                    #Saving to main Power BI Dashboard Dataset
                    df_his=pre.copy()
                    df_his= df_his.drop(columns =['Churn_Probability'])
                    df_his.rename(columns={'Predicted': 'Churn Value'}, inplace = True)
                    his_df = pd.read_csv("../project_main_df.csv")
                    x= list(his_df['Customer ID'])
                    y = df_his['Customer ID'].tolist()
                    res = [ele for ele in x if(ele in y)]
                    if (bool(res)) == True:
                        #st.warning('Customer ID(s) below already exist')
                        his=his_df[his_df['Customer ID'].isin (res)]
                        # st.write(his)
                        his_df= his_df.drop(index= his_df[his_df['Customer ID'].isin(res)].index)
                    his_df=his_df.append(df_his, ignore_index=True)
                    his_df.to_csv("../project_main_df.csv", index = False)
                    st.success("Data saved successfully.. Kindly go to the Dashboard to view customer information")



        elif st.sidebar.checkbox('Select or Update Customer data'):
            if uploaded_file is None:
                st.warning('Please Upload your Data')
            else:
                st.info('Kindly select the Customer Data to Update')
                my_grid = GridOptionsBuilder.from_dataframe(df)
                my_grid.configure_pagination(enabled=True)
                my_grid.configure_default_column(editable=True,groupable=True)
                my_grid.configure_column("Customer ID", editable = False)
                my_grid.configure_column("Married", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Gender", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Female','Male']})
                my_grid.configure_column("Dependents", cellEditor='agRichSelectCellEditor',cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Referred a Friend", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Offer", cellEditor='agRichSelectCellEditor',editable = False)
                my_grid.configure_column("Phone Service", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Multiple Lines", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Internet Service", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Internet Type", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['None','Fiber Optic', 'DSL', 'Cable']})
                my_grid.configure_column("Online Security", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Online Backup", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Device Protection Plan", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Premium Tech Support", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Streaming TV", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Streaming Movies", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Streaming Music", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Unlimited Data", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Contract", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Month-to-Month','One Year', 'Two Year']})
                my_grid.configure_column("Paperless Billing", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("Payment Method", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Credit Card','Bank Withdrawal', 'Mailed Check']})
                my_grid.configure_column("Tenure", editable = False)
                my_grid.configure_column("Distance call", editable = False)
                my_grid.configure_column("Campaign", cellEditor='agRichSelectCellEditor', cellEditorParams={'values':['Yes','No']})
                my_grid.configure_column("age group", editable = False)
                my_grid.configure_column("Total Revenue", editable = False)
                my_grid.configure_column("Monthly Charge", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=2)
                my_grid.configure_column("Age", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=0,)
                my_grid.configure_column("Tenure in Months", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=0)
                my_grid.configure_column("Number of Dependents", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=0)
                my_grid.configure_column("Total Long Distance Charge", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=2)
                my_grid.configure_column("Total Charges", editable = False)
                my_grid.configure_column("Satisfaction Score", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=0)

                sel_option = st.radio('Selection Type', options = ['single', 'multiple'])
                my_grid.configure_selection(selection_mode=sel_option,use_checkbox=True)
                gridoptions = my_grid.build()
                grid_table = AgGrid(df,gridOptions=gridoptions,
                                    enable_enterprise_modules = True,
                                    update_mode= GridUpdateMode.MODEL_CHANGED,
                                    height = 500,
                                    allow_unsafe_jscode=True,
                                    theme = 'material')

                selected_row = grid_table["selected_rows"]
                if st.checkbox('Show data'):
                    try:
                        st.subheader("Selected Data")
                        temp = pd.DataFrame(selected_row)
                        temp['Age'] = temp['Age'].astype(int)
                        temp['Number of Dependents'] = temp['Number of Dependents'].astype(int)
                        temp['Tenure in Months'] = temp['Tenure in Months'].astype(int)
                        temp['Monthly Charge'] = temp['Monthly Charge'].astype(float)
                        temp['Total Charges'] = temp['Total Charges'].astype(float)
                        temp['Total Revenue'] = temp['Total Revenue'].astype(float)
                        temp['Total Long Distance Charges'] = temp['Total Long Distance Charges'].astype(float)
                        #st.dataframe(selected_row)
                        def age_group(row):
                            if row['Age'] <=29:
                                result = 'Under_30'
                            elif row['Age'] >=65:
                                result = 'Senioir_citizen'
                            else:
                                result = 'Young_adult'
                            return result
                        temp['age group'] = temp.apply(age_group, axis = 1)
                        def Depend(row):
                            if row['Number of Dependents'] == 0:
                                result = 'No'
                            else:
                                result = 'Yes'
                            return result
                        temp['Dependents'] = temp.apply(Depend, axis = 1)
                        def internet(row):
                            if row['Internet Type'] == "None":
                                result = 'No'
                            else:
                                result = 'Yes'
                            return result
                        temp['Internet Service'] = temp.apply(internet, axis = 1)
                        def call(row):
                            if row['Total Long Distance Charges'] == 0:
                                result = 'No'
                            else:
                                result = 'Yes'
                            return result
                        temp['Distance call'] = temp.apply(call, axis = 1)
                        def age_group(row):
                            if row['Age'] <=29:
                                result = 'Under_30'
                            elif row['Age'] >=65:
                                result = 'Senioir_citizen'
                            else:
                                result = 'Young_adult'
                            return result
                        temp['age group'] = temp.apply(age_group, axis = 1).astype(str)

                        def promo(row):
                            if row['Campaign'] == 'No':
                                result = 'None'
                            elif row['Tenure in Months'] <=9:
                                result = 'Offer E'
                            elif row['Tenure in Months'] <=23:
                                result = 'Offer D'
                            elif row['Tenure in Months'] <=39:
                                    result = 'Offer C'
                            elif row['Tenure in Months'] <=65:
                                result = 'Offer B'
                            else:
                                    result = 'Offer A'

                            return result
                        temp['Offer'] = temp.apply(promo, axis = 1)
                        def tenure_group(row):
                            if row['Tenure in Months'] <=12:
                                result = '1 year'
                            elif row['Tenure in Months'] <=24:
                                result = '2 years'
                            elif row['Tenure in Months'] <=36:
                                result = '3 years'
                            else:
                                result = '4 years+'
                            return result

                        temp['Tenure'] =temp.apply(tenure_group, axis = 1)

                        temp['Total Charges'] = temp['Tenure in Months'] * temp['Monthly Charge']
                        temp['Total Revenue'] = temp['Total Long Distance Charges'] + temp['Total Charges']

                        st.dataframe(temp)
                    except:
                        st.warning('No Data selected')

                if st.button('Predict'):
                    try:
                        id_df = pd.DataFrame(selected_row)
                        mydf=pd.read_csv("../project_app_df.csv")
                        mydf=mydf.append(id_df, ignore_index = True)
                        mydf.to_csv("../project_app_df.csv", index = False)
                        model = pickle.load(open('RandomForest model.pkl', 'rb'))
                        out_df=mydf.copy()
                        out_df['Age'] = out_df['Age'].astype(int)
                        out_df['Number of Dependents'] = out_df['Number of Dependents'].astype(int)
                        out_df['Tenure in Months'] = out_df['Tenure in Months'].astype(int)
                        out_df['Monthly Charge'] = out_df['Monthly Charge'].astype(float)
                        out_df['Total Charges'] = out_df['Total Charges'].astype(float)
                        out_df['Total Revenue'] = out_df['Total Revenue'].astype(float)
                        out_df['Total Long Distance Charges'] = out_df['Total Long Distance Charges'].astype(float)

                        out_df= out_df.drop(columns=['Customer ID','Gender', 'Age',  'Number of Dependents','Referred a Friend', 'Total Revenue', 'Tenure in Months'])
                        def final():
                            encode_features = [feature for feature  in out_df.columns if out_df[feature ].dtype=='O' and out_df[feature].nunique() == 2]
                            dumy_features = [feature for feature  in out_df.columns if out_df[feature ].dtype=='O' and out_df[feature].nunique() >2]
                            le = LabelEncoder()
                            out_df[encode_features]=out_df[encode_features].apply(le.fit_transform)
                            df_data = pd.get_dummies(out_df, columns=dumy_features, dummy_na=False)
                            final_df= df_data
                            return final_df
                        out_df = final()
                        #st.write(out_df)
                        i = (len(id_df))

                        pre_df = out_df.tail(i)
                        #st.write('data for prediction', pre_df)
                        #pre_df = pre_df.fillna(0)

                        scaler=MinMaxScaler()
                        pre_df=scaler.fit_transform(pre_df)
                        out_pred= model.predict(pre_df)
                        out_prob= model.predict_proba(pre_df)
                        id_df['Predicted'] = out_pred
                        id_df['Churn_Probability'] = (out_prob[:,1]).round(2)


                        def label(row):
                            if row['Predicted'] == 1:
                                result= 'Yes'
                            else:
                                result= 'No'
                            return result
                        id_df['Churn Label'] = id_df.apply(label, axis = 1)
                        id_df['Count'] = 1
                        st.subheader("Prediction on Customer(s) Data" )
                        if i > 1:
                            result = id_df[['Customer ID', 'Churn_Probability', 'Churn Label']]
                            st.write(result)
                        else:
                            if out_pred == 0:
                                st.write('This Customer will **STAY** with the probability of')
                                st.success(out_prob[:,1])
                            else:
                                st.write('**This Customer will **CHURN** with the probability of**')
                                st.warning(out_prob[:,1])
                        st.write('---')

                        pre= pd.DataFrame(id_df)
                        pre['Age'] = pre['Age'].astype(int)
                        pre['Tenure in Months'] = pre['Tenure in Months'].astype(int)
                        pre['Number of Dependents'] = pre['Number of Dependents'].astype(int)
                        pre['Monthly Charge'] = pre['Monthly Charge'].astype(float)
                        pre['Total Charges'] = pre['Total Charges'].astype(float)
                        pre['Total Revenue'] = pre['Total Revenue'].astype(float)
                        pre['Total Long Distance Charges'] = pre['Total Long Distance Charges'].astype(float)
                        st.write('Data to be written',pre)
                        new_df = pd.read_csv("../project_report_df.csv")
                        x= list(new_df['Customer ID'])
                        y = pre['Customer ID'].tolist()
                        res = [ele for ele in x if(ele in y)]
                        if (bool(res)) == True:
                            st.warning('Customer ID(s) below already exist')
                            st.warning('**Warning: The new data below will overwrite existing Customer data**')
                            sub=new_df[new_df['Customer ID'].isin (res)]
                            st.write(sub)
                            new_df= new_df.drop(index= new_df[new_df['Customer ID'].isin(res)].index)

                        new_df=new_df.append(pre, ignore_index=True)
                        new_df.to_csv("../project_report_df.csv", index = False)

                        #Saving to main Power BI Dashboard Dataset
                        df_his=pre.copy()
                        df_his= df_his.drop(columns =['Churn_Probability'])
                        df_his.rename(columns={'Predicted': 'Churn Value'}, inplace = True)
                        his_df = pd.read_csv("../project_main_df.csv")
                        x= list(his_df['Customer ID'])
                        y = df_his['Customer ID'].tolist()
                        res = [ele for ele in x if(ele in y)]
                        if (bool(res)) == True:
                            #st.warning('Customer ID(s) below already exist')
                            his=his_df[his_df['Customer ID'].isin (res)]
                            # st.write(his)
                            his_df= his_df.drop(index= his_df[his_df['Customer ID'].isin(res)].index)
                        his_df=his_df.append(df_his, ignore_index=True)
                        his_df.to_csv("../project_main_df.csv", index = False)
                        st.success("Data added successfully.. ")
                    except:
                        st.write('No data selected')

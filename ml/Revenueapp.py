import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import sklearn



model = pickle.load(open('revenuemodel.pkl', 'rb'))

st.title('Revenue Prediction')




def user_report():
    NumDealsPurchases = st.text_input('NumDealsPurchases')
    Education = st.text_input('Education')
    Kidhome = st.text_input('Kidhome')
    NumWebPurchases = st.text_input('NumWebPurchases')
    NumWebVisitsMonth = st.text_input('NumWebVisitsMonth')
    Year_Birth = st.text_input('Year_Birth')
    Dt_Customer = st.text_input('Dt_Customer')
    MntGoldProds = st.text_input('MntGoldProds')
    MntFruits = st.text_input('MntFruits')
    MntSweetProducts = st.text_input('MntSweetProducts')
    ID = st.text_input('ID')
    MntMeatProducts = st.text_input('MntMeatProducts')
    Income = st.text_input('Income')
    


    user_report_data = {
        'NumDealsPurchases':NumDealsPurchases,
        'Education':Education,	
        'Kidhome':Kidhome,	
        'NumWebPurchases':NumWebPurchases,	
        'NumWebVisitsMonth':NumWebVisitsMonth,	
        'Year_Birth':Year_Birth,	
        'Dt_Customer':Dt_Customer,	
        'MntGoldProds':MntGoldProds,
        'MntFruits':MntFruits,	
        'MntSweetProducts':MntSweetProducts,	
        'ID':ID,	
        'MntMeatProducts':MntMeatProducts,	
        'Income':Income	
      
    }   
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


user_data = user_report() 

if st.button("predict"):
     model.predict(user_data)
     st.write(model.predict(user_data))
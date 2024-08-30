from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

def home(request):
    return render(request, "Home.html")

def predict(request):
    return render(request, "predict.html")

def result(request):
    df = pd.read_csv('D:\downloads\Revenue_prediction\ml\marketing_data_1 - marketing_data_1 (1).csv')
    df['Income'] = df['Income'].str.replace('$','')
    df.dropna(inplace=True)

    le = LabelEncoder()
    df['Education'] = le.fit_transform(df['Education'])
    df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
    df['Dt_Customer'] = df['Dt_Customer'].astype('category').cat.codes
    df['Country'] = le.fit_transform(df['Country'])
    df['Income'] = df['Income'].str.replace(',', '')

    from sklearn.decomposition import PCA
    selected_features = df.iloc[:, :28]
    pca = PCA(n_components=15)
    principal_components = pca.fit_transform(selected_features)
    loadings = pca.components_[0]
    loading_df = pd.DataFrame({'Feature': selected_features.columns, 'Loading': loadings})

    loading_df['Abs_Loading'] = loading_df['Loading'].abs()
    loading_df = loading_df.sort_values(by='Abs_Loading', ascending=False)

    useful_cols = ['NumDealsPurchases', 'Education', 'Kidhome', 'NumWebPurchases', 'NumWebVisitsMonth', 'Year_Birth', 'Dt_Customer','MntGoldProds', 'MntFruits', 'MntSweetProducts', 'ID', 'MntMeatProducts', 'Income', 'MntWines']
    df = df[useful_cols]

    X = df.iloc[:,:-1].values
    Y = df.iloc[:,-1].values

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 1, test_size = 0.3)

    model = LinearRegression()
    model.fit(x_train, y_train)

    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])
    var6 = float(request.GET['n6'])
    var7 = float(request.GET['n7'])
    var8 = float(request.GET['n8'])
    var9 = float(request.GET['n9'])
    var10 = float(request.GET['n10'])
    var11 = float(request.GET['n11'])
    var12 = float(request.GET['n12'])
    var13 = float(request.GET['n13'])

    pred = model.predict(np.array([var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13]).reshape(1,-1))
    pred = round(pred[0])

    Revenue = "The Predicted Revenue is RS: "+str(pred)

    
    return render(request, "predict.html", {"result2":Revenue})
    

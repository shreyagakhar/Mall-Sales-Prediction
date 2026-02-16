import streamlit as st
import pickle
import pandas as pd
import numpy as np

df= pd.read_csv('Mall_df.csv')
df= df.set_index('date')
df.index = pd.to_datetime(df.index).normalize()
df.columns= ['Past Sales']

tes= pickle.load(open('tes.pkl', 'rb'))
arima= pickle.load(open('arima.pkl', 'rb'))
sarima= pickle.load(open('sarima.pkl', 'rb'))

st.title('Predicting Overall Mall Sales')

st.write('Select the number of days you need to forecast sales for:')
days= st.slider('Number of days',1,30,1)

selection= st.selectbox('Select the Model:', ['Holt-Winters', 'ARIMA', 'SARIMA'], None)

if selection=='Holt-Winters':
    model= tes
elif selection=='ARIMA':
    model= arima
elif selection=='SARIMA':
    model= sarima
else:
    model= None


if st.button("Generate Forecast"):
    if model== None:
        st.error('Pleae select a model to generate a prediction')
    else:
        forecast = model.forecast(days)
        forecast_df = pd.DataFrame({"Forecast": forecast})
        new_df = pd.concat([df, forecast_df], axis=1)
        st.line_chart(new_df, x_label='Dates', y_label='Sales', color=['blue', 'orange'])
        st.dataframe(forecast_df)




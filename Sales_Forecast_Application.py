## Importing the packages 
import streamlit as st
import pandas as pd
import numpy as np
import pickle 

# Setting the title with an enhanced UI and metallic background
def set_title():
    st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #434343, #000000);
            color: white;
        }
        .main-title {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            color: #FFD700;
            text-shadow: 3px 3px 5px rgba(255, 255, 255, 0.3);
        }
    </style>
    <h1 class='main-title'>Product Sales Analysis & Time Series Forecast</h1>
    <hr>
    """, unsafe_allow_html=True)

# Displaying the historical sales trend
def display_historical_sales(historical_data):
    st.subheader("📊 Sales Trend")
    historical_data = historical_data.loc[:'2019-04-01']
    st.line_chart(historical_data)

# Displaying the historical data and forecast data in one chart
def display_combined_chart(historical_data, forecast_data):
    st.subheader("📊 Combined Sales Trend and Forecast")

    # Filter historical data from 01-March-2019 to 01-April-2019
    historical_data = historical_data.loc['2019-03-01':'2019-04-02']

    # Combine historical and forecast data for plotting
    combined_df = pd.concat([
        historical_data.rename(columns={'Sales': 'Historical Sales'}),
        forecast_data.rename(columns={'predicted_mean': 'Forecasted Sales'})
    ], axis=1)

    # Format dates in the index consistently
    combined_df.index = pd.to_datetime(combined_df.index).strftime('%Y-%m-%d')

    # Plot combined data
    st.line_chart(combined_df)



# Taking user inputs in one row
def get_user_inputs():
    st.subheader("📋 Enter Forecast Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        num = st.number_input('📅 Forecast Days', min_value=1, step=1)
    with col2:
        ex_flag = st.radio('📈 Add Exogenous Variables?', ['No', 'Yes'])
    
    if ex_flag == 'Yes':
        with col3:
            ex_holiday = st.text_input('🏖 Holidays (1=Yes, 0=No)', placeholder='1,1,0,1,1')
            ex_discount = st.text_input('💲 Discounts (1=Yes, 0=No)', placeholder='1,0,0,1,1')
    else:
        ex_holiday, ex_discount = None, None
    
    return num, ex_flag, ex_holiday, ex_discount

# Loading the SARIMA model for forecasting without exogenous variables
def forecast_without_exog(num):
    with open('sarima.pkl', 'rb') as file:
        sarima = pickle.load(file)
    return sarima.forecast(num)

# Checking and getting exogenous inputs
def get_exogenous_inputs(ex_flag, num, ex_holiday, ex_discount):
    if ex_flag == 'Yes' and ex_holiday and ex_discount:
        try:
            holidays = [int(x) for x in ex_holiday.split(',')]
            discounts = [int(x) for x in ex_discount.split(',')]
            
            if len(holidays) == num and len(discounts) == num:
                return np.column_stack((holidays, discounts))
            else:
                st.error("⚠️ Exogenous values must match forecast days")
                return None
        except:
            st.error("⚠️ Enter valid binary values (0 or 1)")
            return None
    return None

# Loading the SARIMAX model for exogenous forecasting
def forecast_with_exog(num, exog):
    with open('sarimax.pkl', 'rb') as file:
        sarimax = pickle.load(file)
    return sarimax.forecast(num, exog=exog)

def display_forecast_table(sales):
    sales_df = sales.reset_index()
    sales_df.columns = ["Date", "Forecasted Sales"]
    st.subheader("📋 Forecasted Sales Table")
    st.dataframe(sales_df)



# Main execution flow
def main():
    set_title()

    # Load historical data
    dftrainfe = pd.read_csv('TRAIN.csv')
    historical_data = dftrainfe.pivot_table(index='Date', values=['Sales'], aggfunc='sum')

    # Display historical sales trend
    display_historical_sales(historical_data)

    # Display user inputs
    num, ex_flag, ex_holiday, ex_discount = get_user_inputs()
    exog = get_exogenous_inputs(ex_flag, num, ex_holiday, ex_discount)

    if st.button('🚀 Predict Sales'):
        if exog is None:
            forecast_data = pd.DataFrame(forecast_without_exog(num))
        else:
            forecast_data = pd.DataFrame(forecast_with_exog(num, exog))

        # Display combined chart
        display_combined_chart(historical_data, forecast_data)
        display_forecast_table(forecast_data)

if __name__ == "__main__":
    main()

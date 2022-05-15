import streamlit as st
import matplotlib.pyplot as plt
from googletrans import Translator
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import pandas_datareader as datas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'



# Стиль
def set_pub():
    plt.rc('font', weight='bold')  # bold fonts are easier to see
    plt.rc('grid', c='0.5', ls='-', lw=0.5)
    plt.rc('figure', figsize=(10, 8))
    plt.style.use('bmh')
    plt.rc('lines', linewidth=1.3, color='b')


# Скользящие средние
def moving_avarage(name, start_data, end_data):
    data = yf.download(name, start_data, end_data)
    short_ma = 5
    long_ma = 12
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    sr_sell = 0.7
    sr_buy = 0.3
    data['MA' + str(short_ma)] = data["Close"].rolling(short_ma).mean()
    data['MA' + str(long_ma)] = data["Close"].rolling(long_ma).mean()
    data['return'] = data['Close'].pct_change()
    data['Up'] = np.maximum(data['Close'].diff(), 0)
    data['Down'] = np.maximum(-data['Close'].diff(), 0)
    data['RS'] = data['Up'].rolling(rsi_period).mean() / data['Down'].rolling(rsi_period).mean()
    data['RSI'] = 100 - 100 / (1 + data['RS'])
    data['S&R'] = (data['Close']) / (10 ** np.floor(np.log10(data['Close']))) % 1

    start = max(long_ma, rsi_period)
    data['MACD_signal'] = 2 * (data['MA' + str(short_ma)] > data['MA' + str(long_ma)]) - 1
    data['RSI_signal'] = 1 * (data['RSI'] < rsi_oversold) - 1 * (data['RSI'] > rsi_overbought)
    data['S&R_signal'] = 1 * (data['S&R'] < sr_buy) - 1 * (data['S&R'] > 0.7)

    BnH_return = np.array(data['return'][start + 1:])
    MACD_return = np.array(data['return'][start + 1:]) * np.array(data['MACD_signal'][start:-1])
    RSI_return = np.array(data['return'][start + 1:]) * np.array(data['RSI_signal'][start:-1])
    SnR_return = np.array(data['return'][start + 1:]) * np.array(data['S&R_signal'][start:-1])

    BnH = np.prod(1 + BnH_return) ** (252 / len(BnH_return))
    MACD = np.prod(1 + MACD_return) ** (252 / len(MACD_return))
    RSI = np.prod(1 + RSI_return) ** (252 / len(RSI_return))
    SnR = np.prod(1 + SnR_return) ** (252 / len(SnR_return))

    BnH_risk = np.std(BnH_return) * (252) ** (1 / 2)
    MACD_risk = np.std(MACD_return) * (252) ** (1 / 2)
    RSI_risk = np.std(RSI_return) * (252) ** (1 / 2)
    SnR_risk = np.std(SnR_return) * (252) ** (1 / 2)

    st.write('Доходность риск стратегии Buy-and-hold ' + str(round(BnH * 100, 2)) + '% и ' + str(
        round(BnH_risk * 100, 2)) + '%')
    st.write('Доходность риск стратегии скользящих средних ' + str(round(MACD * 100, 2)) + '% и ' + str(
        round(MACD_risk * 100, 2)) + '%')
    st.write('Доходность риск стратегии RSI ' + str(round(RSI * 100, 2)) + '% и ' + str(round(RSI_risk * 100, 2)) + '%')
    st.write('Доходность риск стратегии поддержки и сопротивления ' + str(round(SnR * 100, 2)) + '% и ' + str(
        round(SnR_risk * 100, 2)) + '%')

    return st.dataframe(data), st.line_chart(data[['Close', 'MA12', 'MA5']])


# Reinforcement Learning
def Reinforcement_learning(name):
    pass

def Machine_learning(name, today):

    # Get the stock data, starting from 2000-01-01 to today
    df = datas.DataReader(name, 'yahoo', '2000-01-01', today)
    # For the prediction we only need the column/variable "Adj Close"
    df = df[['Adj Close']]

    # Creating a variable "n" for predicting the amount of days in the future
    # We predict the stock price 30 days in the future
    n = 30

    # Create another column "Prediction" shifted "n" units up
    df['Prediction'] = df[['Adj Close']].shift(-n)
    # We shifted the data up 30 rows, so that for every date we have the actual price ("Adj Close") and the predicted price 30 days into the future ("Prediction")
    # Therefore the last 30 rows of the column "Prediction" will be empty or contain the value "NaN"

    # Creating independent data set "X"
    # For the independent data we dont need the column "Prediction"
    X = df.drop(['Prediction'], axis = 1)
    # Convert the data into a numpy array
    X = np.array(X)
    # Remove the last "n" rows
    X = X[:-n]

    # Create the dependent data set "Y"
    # For the dependent data we need the column "Prediction"
    Y = df['Prediction']
    # Convert the data into a numpy array
    Y = np.array(Y)
    # Remove the last "n" rows
    Y = Y[:-n]

    # Split the data into 80% train data and 20 % test data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # Create Linear Regression Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)

    # Set "forecast" to the last 30 rows of the original data set from "Adj Close" column
    # We dont need the column 'Prediction'
    # Convert the data into a numpy array
    # We want the last 30 rows
    forecast = np.array(df.drop(['Prediction'], axis=1))[-n:]

    # Print the predictions for the next "n" days
    # "lr_prediction" contains the price values, which the Linear Regression Model has predicted for the next "n" (30) days
    lr_prediction = lr.predict(forecast)

    # Now we save the predictions in a DataFrame called "predictions"
    predictions = pd.DataFrame(lr_prediction, columns=['Prediction'])
    # "predictions" has 1 column with the predicted values
    # However to plot the value we need another variable/column, which indicates the respective date

    # Therefore we replace the index of the initial data set with simple sequential numbers and save the old index ("DatetimeIndex") as a variable "Date"
    df = df.reset_index()

    # From "Date" we need the to get the last value which is the latest date and add 1 day, because that's the date when our predictions start
    d = df['Date'].iloc[-1]
    d = d + relativedelta(days=+ 1)

    # Now we make a list with the respective daterange, beginning from the startdate of our predictions and ending 30 days after
    datelist = pd.date_range(d, periods=30).tolist()
    # We add the variable to our Dataframe "predictions"
    predictions['Date'] = datelist
    # Now we have a Dataframe with our predicted values and the correspondig dates

    # Save the date of today 6 months ago, by subtracting 6 months from the date of today
    six_months = date.today() - relativedelta(months=+6)
    six_months = six_months.strftime('%Y-%m-%d')

    # Get the data for plotting
    df = datas.DataReader(name, 'yahoo', six_months, today)
    df = df.reset_index()

    # Plotting the chart
    fig = go.Figure()
    # Add the data from the first stock
    fig.add_trace(go.Scatter(
        x=df.Date,
        y=df['Adj Close'],
        name=f'{name} stock',
        line_color='deepskyblue',
        opacity=0.9))

    # Add the data from the predictions
    fig.add_trace(go.Scatter(
        x=predictions.Date,
        y=predictions['Prediction'],
        name=f'Prediction',
        line=dict(color='red', dash='dot'),
        opacity=0.9))

    fig.update_layout(title=f'Stock Forecast of {name} Stock for the next 30 days',
                      yaxis_title='Adjusted Closing Price',
                      xaxis_tickfont_size=14,
                      yaxis_tickfont_size=14)

    st.write(fig)


def get_fundamental_data(name):
    metric = ['P/B',
              'P/E',
              'Forward P/E',
              'PEG',
              'Debt/Eq',
              'EPS (ttm)',
              'Dividend %',
              'ROE',
              'ROI',
              'EPS Q/Q',
              'Insider Own'
              ]
    df = pd.DataFrame(index=[name], columns=metric)
    def fundamental_metric(soup, metric):
        return soup.find(text=metric).find_next(class_='snapshot-td2').text
    for symbol in df.index:
        try:
            url = ("http://finviz.com/quote.ashx?t=" + name.lower())
            req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
            response = urlopen(req)
            soup = BeautifulSoup(response)
            for m in df.columns:
                df.loc[symbol,m] = fundamental_metric(soup,m)
        except Exception as e:
            print (symbol, 'not found')
    st.write(df)


# Мнения аналитиков
def analysis(name, today):
    # Save the date of today 6 months ago, by subtracting 6 months from the date of today
    six_months = date.today() - relativedelta(months=+6)
    six_months = six_months.strftime('%Y-%m-%d')

    data = yf.Ticker(name)
    # Save the Analyst Recommendations in "rec"
    Analitics_rec = data.recommendations
    if Analitics_rec.empty:
        st.write("> Unfortunately, there are no recommendations by analysts provided for your chosen stock!")
    # The DataFrame "rec" has 4 columns: "Firm", "To Grade", "From Grade" and "Action"
    # The index is the date ("DatetimeIndex")
    # Now we select only those columns which have the index(date) from "six months" to "today"
    else:
        Analitics_rec = Analitics_rec.loc[six_months:today, ]
        st.write(Analitics_rec)
        # Replace the index with simple sequential numbers and save the old index ("DatetimeIndex") as a variable "Date"
        rec = Analitics_rec.reset_index()

        # For our analysis we don't need the variables/columns "Firm", "From Grade" and "Action", therefore we delete them
        rec.drop(['Firm', 'From Grade', 'Action'], axis=1, inplace=True)

        # We change the name of the variables/columns
        rec.columns = (['date', 'grade'])

        # Now we add a new variable/column "value", which we give the value 1 for each row in order to sum up the values based on the contents of "grade"
        rec['value'] = 1

        # Now we group by the content of "grade" and sum their respective values
        rec = rec.groupby(['grade']).sum()
        # The DataFrame "rec" has now 1 variable/column which is the value, the index are the different names from the variable "grade"
        # However for the plotting we need the index as a variable
        rec = rec.reset_index()

        # For the labels we assign the content/names of the variable "grade" and for the values we assign the content of "values"
        st.write(f'Analyst Recommendations of {name} Stock from {six_months} to {today}')
        fig1, ax1 = plt.subplots(figsize=(16,9))
        fig1.patch.set_facecolor('#0e1117')
        ax1.pie(rec.value, labels=rec.grade, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.legend(labels=rec.grade)

        st.pyplot(fig1)




def main():
    sp500_list = pd.read_csv('SP500_list.csv')
    ticker = st.selectbox('Select the ticker if present in the S&P 500 index', sp500_list['Symbol'], index=26).upper()
    pivot_sector = True
    checkbox_noSP = st.checkbox('Select this box to write the ticker (if not present in the S&P 500 list). \
                                Deselect to come back to the S&P 500 index stock list')
    if checkbox_noSP:
        ticker = st.text_input('Write the ticker (check it in yahoo finance)', 'MN.MI').upper()
        pivot_sector = False

    # Задаем диапазон дат
    start = st.text_input('Enter the start date in yyyy-mm-dd format:', '2021-01-01')
    today = date.today()
    today = today.strftime('%Y-%m-%d')
    end = st.text_input('Enter the end date in yyyy-mm-dd format:', today)

    try:
        ticker_meta = yf.Ticker(ticker)

        series_info = pd.Series(ticker_meta.info)
        series_info = series_info.loc[
            ['symbol', 'shortName', 'financialCurrency', 'longBusinessSummary', 'sector', 'country',
             'exchangeTimezoneName', 'currency', 'quoteType']]
        # test = series_info.astype(str)
        # st.dataframe(test)

        # Основная инфа о компании
        translator = Translator()
        st.write(translator.translate(text=series_info[3], src='en', dest='ru'))

        col1, col2, col3 = st.columns(3)
        col1.metric("Тикер", series_info[0])
        col2.metric("Название", series_info[1])
        col3.metric("Сектор", series_info[4])

        col4, col5, col6 = st.columns(3)
        col1.metric("Страна", series_info[5])
        col2.metric("Биржа", series_info[6])
        col3.metric("Валюта", series_info[7])

        moving_avarage(ticker, start, end)
        Machine_learning(ticker, today)
        get_fundamental_data(ticker)
        analysis(ticker, today)

    except KeyError:
        st.write('Try to input correct name')



if __name__ == '__main__':
    main()

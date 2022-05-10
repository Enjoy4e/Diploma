import streamlit as st
import matplotlib.pyplot as plt
from googletrans import Translator
import numpy as np
import pandas as pd
import yfinance as yf

# Считываем данные
sp500_list = pd.read_csv('SP500_list.csv')
ticker = st.selectbox('Select the ticker if present in the S&P 500 index', sp500_list['Symbol'], index=30).upper()
pivot_sector = True
checkbox_noSP = st.checkbox('Select this box to write the ticker (if not present in the S&P 500 list). \
                            Deselect to come back to the S&P 500 index stock list')
if checkbox_noSP:
    ticker = st.text_input('Write the ticker (check it in yahoo finance)', 'MN.MI').upper()
    pivot_sector = False

# Задаем диапазон дат
start = st.text_input('Enter the start date in yyyy-mm-dd format:', '2021-01-01')
end = st.text_input('Enter the end date in yyyy-mm-dd format:', '2022-01-01')

ticker_meta = yf.Ticker(ticker)

series_info = pd.Series(ticker_meta.info)
series_info = series_info.loc[['symbol', 'shortName', 'financialCurrency', 'longBusinessSummary', 'sector', 'country',
                               'exchangeTimezoneName', 'currency', 'quoteType']]
#test = series_info.astype(str)
#st.dataframe(test)

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
    data['MA'+str(short_ma)] = data["Close"].rolling(short_ma).mean()
    data['MA'+str(long_ma)] = data["Close"].rolling(long_ma).mean()
    data['return'] = data['Close'].pct_change()
    data['Up'] = np.maximum(data['Close'].diff(), 0)
    data['Down'] = np.maximum(-data['Close'].diff(), 0)
    data['RS'] = data['Up'].rolling(rsi_period).mean()/data['Down'].rolling(rsi_period).mean()
    data['RSI'] = 100 - 100/(1+data['RS'])
    data['S&R'] = (data['Close'])/(10**np.floor(np.log10(data['Close'])))%1

    start = max(long_ma, rsi_period)
    data['MACD_signal'] = 2*( data['MA'+str(short_ma)]>data['MA'+str(long_ma)]) - 1
    data['RSI_signal'] = 1*(data['RSI'] < rsi_oversold) - 1 * (data['RSI'] > rsi_overbought)
    data['S&R_signal'] = 1*(data['S&R'] < sr_buy) - 1 * (data['S&R'] > 0.7)

    BnH_return = np.array(data['return'][start+1:])
    MACD_return = np.array(data['return'][start+1:])*np.array(data['MACD_signal'][start:-1])
    RSI_return = np.array(data['return'][start+1:])*np.array(data['RSI_signal'][start:-1])
    SnR_return = np.array(data['return'][start+1:])*np.array(data['S&R_signal'][start:-1])

    BnH = np.prod(1+BnH_return)**(252/len(BnH_return))
    MACD = np.prod(1+MACD_return)**(252/len(MACD_return))
    RSI = np.prod(1+RSI_return)**(252/len(RSI_return))
    SnR = np.prod(1+SnR_return)**(252/len(SnR_return))

    BnH_risk = np.std(BnH_return)*(252)**(1/2)
    MACD_risk = np.std(MACD_return)*(252)**(1/2)
    RSI_risk = np.std(RSI_return)*(252)**(1/2)
    SnR_risk = np.std(SnR_return)*(252)**(1/2)

    st.write('Доходность риск стратегии Buy-and-hold '+str(round(BnH*100, 2))+'% и '+str(round(BnH_risk*100,2))+'%')
    st.write('Доходность риск стратегии скользящих средних '+str(round(MACD*100, 2))+'% и '+str(round(MACD_risk*100,2))+'%')
    st.write('Доходность риск стратегии RSI '+str(round(RSI*100, 2))+'% и '+str(round(RSI_risk*100,2))+'%')
    st.write('Доходность риск стратегии поддержки и сопротивления '+str(round(SnR*100, 2))+'% и '+str(round(SnR_risk*100,2))+'%')

    return st.dataframe(data), st.line_chart(data[['Close','MA12', 'MA5']])

def analysis(name):
    stock = yf.Ticker(name)
    stock_analitics = stock.recommendations
    if stock_analitics is None:
        st.write('На данный момент прогнозов от аналитиков - нет')
    else:
        st.write(stock_analitics)
        stock_analitics.columns = stock_analitics.columns.str.replace(" ", "_")
       # print(stock_analitics.columns.tolist)
        st.write(stock_analitics.To_Grade.value_counts())


def additive_regression_model(name):
    pass

def plot_data():
    pass



def main():
    moving_avarage(ticker, start, end)
    analysis(ticker)

if __name__ == '__main__':
    main()

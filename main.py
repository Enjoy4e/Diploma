import time
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

    data['MA' + str(short_ma)] = data["Close"].rolling(short_ma).mean()
    data['MA' + str(long_ma)] = data["Close"].rolling(long_ma).mean()
    data['return'] = data['Close'].pct_change()
    data['Up'] = np.maximum(data['Close'].diff(), 0)
    data['Down'] = np.maximum(-data['Close'].diff(), 0)
    data['RS'] = data['Up'].rolling(rsi_period).mean() / data['Down'].rolling(rsi_period).mean()
    data['RSI'] = 100 - 100 / (1 + data['RS'])

    start = max(long_ma, rsi_period)
    data['MACD_signal'] = 2 * (data['MA' + str(short_ma)] > data['MA' + str(long_ma)]) - 1
    data['RSI_signal'] = 1 * (data['RSI'] < rsi_oversold) - 1 * (data['RSI'] > rsi_overbought)

    MACD_return = np.array(data['return'][start + 1:]) * np.array(data['MACD_signal'][start:-1])
    RSI_return = np.array(data['return'][start + 1:]) * np.array(data['RSI_signal'][start:-1])

    BnH = data['Close'][-1]*100/data['Close'][0]
    MACD = np.prod(1 + MACD_return) ** (252 / len(MACD_return))
    RSI = np.prod(1 + RSI_return) ** (252 / len(RSI_return))

    data['RSI_sold'] = np.nan
    data['RSI_buy'] = np.nan
    data['buy_signal'] = np.nan
    data['sold_signal'] = np.nan
    for i in range(1,len(data)):
        if data['RSI_signal'][i] == -1:
            data['RSI_sold'][i] = data['RSI'][i]
            data['sold_signal'][i] = data['Close'][i]
        elif data['RSI_signal'][i] == 1:
            data['RSI_buy'][i] = data['RSI'][i]
            data['buy_signal'][i] = data['Close'][i]
    if round(RSI * 100, 2) > 100:
        st.success('Доходность риск стратегии RSI ' + str(round(RSI * 100, 2)) + "%")
    else:
        st.error('Доходность риск стратегии RSI ' + str(round(RSI * 100, 2)) + "%")
    ##График RSI
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, sharex=True, figsize=(13, 9))
    fig.suptitle('RSI Стратегия')
    ## Покупка продажа на основном графике:
    axs[0].scatter(data.index, data['buy_signal'], color='green', marker='^', alpha=1)
    axs[0].scatter(data.index, data['sold_signal'], color='red', marker='v', alpha=1)
    axs[0].plot(data['Adj Close'], alpha=0.8)
    axs[0].grid()
    ## Покупка продажа на графике RSI
    axs[1].scatter(data.index, data['RSI_buy'], color='green', marker='^', alpha=1)
    axs[1].scatter(data.index, data['RSI_sold'], color='red', marker='v', alpha=1)
    axs[1].plot(data['RSI'], alpha=0.8)
    axs[1].grid()
    st.pyplot(fig)

    ##Скользящие средние
    if round(MACD * 100, 2) > 100:
        st.success('Доходность риск стратегии скользящих средних ' + str(round(MACD * 100, 2)) + "%")
    else:
        st.error('Доходность риск стратегии скользящих средних ' + str(round(MACD * 100, 2)) + "%")

    fig1 = plt.figure(figsize=(14, 8))
    fig1.suptitle('MACD Стратегия')
    plt.style.use('dark_background')
    plt.plot(data.index, data['MA' + str(short_ma)], label='MA5', color='blue')
    plt.plot(data.index, data['MA' + str(long_ma)], label='MA12', color='red')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    st.pyplot(fig1)

    if round(BnH,2) > 100:
        st.success('Доходность риск стратегии Buy-and-hold ' + str(round(BnH, 2)) + "%")
    else:
        st.error('Доходность риск стратегии Buy-and-hold ' + str(round(BnH, 2)) + "%")

    return st.dataframe(data), st.line_chart(data[['Close', 'MA12', 'MA5']])


# Reinforcement Learning
def LSTM(name):
    pass

def Machine_learning(name, today):
    # Получение данных
    df = datas.DataReader(name, 'yahoo', '2000-01-01', today)
    # Закрытие
    df = df[['Adj Close']]

    # Количество дней для которых будем пытаться определить котировки
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

    fig.update_layout(title=f'Прогнорование цен акций {name} на следующий месяц',
                      yaxis_title='Adjusted Closing Price',
                      xaxis_tickfont_size=14,
                      yaxis_tickfont_size=14)

    st.write(fig)

def fundamental_metric(soup, metric):
    return soup.find(text=metric).find_next(class_='snapshot-td2').text


def get_fundamental_data(name):
    metric = ['P/B',
              'P/E',
              'Forward P/E',
              'PEG',
              'Debt/Eq',
              'Dividend %',
              'ROE',
              'ROI',
              'EPS Q/Q',
              ]
    df = pd.DataFrame(index=[name], columns=metric)
    try:
        url = ("http://finviz.com/quote.ashx?t=" + name.lower())
        req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
        response = urlopen(req)
        soup = BeautifulSoup(response)
        for m in df.columns:
            df.loc[name,m] = fundamental_metric(soup,m)
        st.write(df)
        df1 = df
        df1['Dividend %'] = df1['Dividend %'].str.replace('%', '')
        df1['ROE'] = df1['ROE'].str.replace('%', '')
        df1['ROI'] = df1['ROI'].str.replace('%', '')
        df1['EPS Q/Q'] = df1['EPS Q/Q'].str.replace('%', '')
        df1 = df1.apply(pd.to_numeric, errors='coerce')

        # Цена акции/прибыль на акцию. EPS - Отношение чистой прибыли на количество акций в обращении.
        if (df1['P/E'].astype(float) < 0).any() == False:
            st.write(f'Показатель P/E:')
            st.error('Компания с отрицательной прибылью')
        elif (df1['P/E'].astype(float) > 20).any() == True:
            st.write(f'Показатель P/E: {df1["P/E"].iloc[0]} %')
            st.warning('Компания может быть переоценена')
        else:
            st.write(f'Показатель P/E: {df1["P/E"].iloc[0]} %')
            st.success('Компанию стоит рассматривать к покупке')

        # Капитализация/Балансовую стоимость компании(чистые активы)
        if (df1['P/B'].astype(float) < 0).any() == True:
            st.write(f'Показатель P/B: {df1["P/B"].iloc[0]} %')
            st.error(
                'У компании долгов больше, чем собственных активов, может привести к банкротству, не стоит рассматривать такие компании для покупки')
        elif (df1['P/B'].astype(float) > 0).any() == True and (df1['P/B'].astype(float) < 1).any() == True:
            st.write(f'Показатель P/B: {df1["P/B"].iloc[0]} %')
            st.success(
                'Капитализация компании меньше ее собственного капитала, акции вы можете приобрести ее со скидкой')
        elif (df1['P/B'].astype(float) == 1).any() == True:
            st.write(f'Показатель P/B: {df1["P/B"].iloc[0]} %')
            st.warning('Акция оценена справедливо')
        else:
            st.write(f'Показатель P/B: {df1["P/B"].iloc[0]} %')
            st.warning('Капитализация компании большее ее собственного капитала, за акции вы переплачиваете')

        # Дивидендная доходность
        if (df1['Dividend %']).any() == False:
            st.write('Дивидендная доходность:')
            st.warning('Компания не выплачивает дивиденды')
        else:
            st.write('Дивидендная доходность:')
            st.success(f'Дивидендная доходность составляет:{df1["Dividend %"].iloc[0]}%')

        # Эффективност вложений. Сколько инвестор получает за вложенный рубль (доходы-затраты/затраты)*100
        if (df1['ROI'].astype(float) > 100).any() == True:
            st.write(f'Показатель ROI: {df1["ROI"].iloc[0]} %')
            st.success('Бизнес окупился и приносит прибыль')
        else:
            st.write(f'Показатель ROI: {df1["ROI"].iloc[0]} %')
            st.error('Инвестиции не окупились: компания вкладывает больше, чем получает')

        # Рентабельность собственного капитала (Годовая чистая прибыль/среднегодовой собственный капитал)*100
        # По сути процентная ставка
        if (df1['ROE'].astype(float) > 17).any() == True:
            st.write(f'Показатель ROE: {df1["ROE"].iloc[0]} %')
            st.success('Компания привлекательная для покупки, так как может дать прибыль больше, чем дает банковский вклад')
        else:
            st.write(f'Показатель ROE: {df1["ROE"].iloc[0]} %')
            st.warning('Низкая прибыль')

        # Выгода от покупки (P/E)/EPS
        if (df1['PEG'].astype(float)).any() == False:
            st.write(f'Показатель PEG:')
            st.error('Компания с отрицательной прибылью')
        elif (df1['PEG'].astype(float) > 1).any() == True:
            st.write(f'Показатель PEG: {df1["PEG"].iloc[0]} ')
            st.warning('Компания переоценена')
        elif (df1['PEG'].astype(float) == 1).any() == True:
            st.write(f'Показатель PEG: {df1["PEG"].iloc[0]} ')
            st.warning('Компания оценена справедливо')
        else:
            st.write(f'Показатель PEG: {df1["PEG"].iloc[0]} ')
            st.success('Компанию стоит рассматривать к покупке')

        # Изменение прибыли на акцию
        if (df1['EPS Q/Q'].astype(float) > 17).any() == True:
            st.write(f'Показатель EPS Q/Q: {df1["EPS Q/Q"].iloc[0]} ')
            st.success('Компания привлекательная для покупки, так как может дать прибыль больше, чем дает банковский вклад')
        else:
            st.write(f'Показатель EPS Q/Q: {df1["EPS Q/Q"].iloc[0]} ')
            st.warning('Низкая прибыль')
    except Exception as e:
        st.error(name, 'Данные по компании не найдены')



# Мнения аналитиков
def analysis(name, today):
    # Save the date of today 6 months ago, by subtracting 6 months from the date of today
    six_months = date.today() - relativedelta(months=+6)
    six_months = six_months.strftime('%Y-%m-%d')

    data = yf.Ticker(name)
    # Save the Analyst Recommendations in "rec"
    Analitics_rec = data.recommendations
    try:
        if Analitics_rec.empty:
            st.write("За заданный промежуток времени нет ни одной рекомендации от аналитика")
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
            st.write(f'Рекомендации аналитиков на акции {name} от {six_months} по {today}')
            fig1, ax1 = plt.subplots(figsize=(16,9))
            fig1.patch.set_facecolor('#0e1117')
            ax1.pie(rec.value, labels=rec.grade, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.legend(labels=rec.grade)

            st.pyplot(fig1)

    except AttributeError:
        st.error("Нет еще ни одной рекомендации от аналитиков")

def main():
    sp500_list = pd.read_csv('SP500_list.csv')
    ticker = st.selectbox('Выберите тикер компании из списка компаний S&P500', sp500_list['Symbol'], index=45).upper()
    pivot_sector = True
    checkbox_noSP = st.checkbox('Выберите данный пункт и введите название тикера компании (если она не представлена в списке S&P500). \
                                Отменить выбор - вернуться к списку S&P500', key = 1)
    if checkbox_noSP:
        ticker = st.text_input('Введите название тикера (лучше проверить на сайте Yahoo Finance)', 'MN.MI').upper()
        pivot_sector = False

    # Задаем диапазон дат
    kek = True
    kek1 = True
    st.subheader('Задаем диапазон для анализа')
    start = st.text_input('Введите стартовую дату в формате yyyy-mm-dd:', '2021-01-01')
    try:
        time.strptime(start, "%Y-%m-%d")
    except ValueError:
        kek = False
        st.error('Введите корректную начальную дату!')
    today = date.today()
    today = today.strftime('%Y-%m-%d')
    end = st.text_input('Введите конечную дату в формте yyyy-mm-dd:', today)
    try:
        time.strptime(end, "%Y-%m-%d")
    except ValueError:
        kek1 = False
        st.error('Введите корректную конечную дату!')
    if kek1 == True and kek == True and start < end:
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
            with st.expander('О компании:'):
                st.caption(translator.translate(text=series_info[3], src='en', dest='ru'))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Тикер")
                st.subheader(series_info[0])
            with col2:
                st.write("Полное название")
                st.subheader(series_info[1])
            with col3:
                st.write("Сектор")
                st.subheader(series_info[4])

            col4, col5, col6 = st.columns(3)
            with col4:
                st.write("Страна")
                st.subheader(series_info[5])
            with col5:
                st.write("Биржа")
                st.subheader(series_info[6])
            with col6:
                st.write("Валюта")
                st.subheader(series_info[7])

            st.markdown("---")
            checkbox_moving_avarage = st.checkbox('Отобразить временной график и технический анализ', key = 2)
            try:
                if checkbox_moving_avarage:
                        with st.expander('Описание:'):
                            st.caption('- RSI (индекс относительной силы), определяющий (уровень падения цены, абсолютную величину ее роста)' )
                            st.caption('- MACD – это, по своему, уникальный индикатор, поскольку сочетает в себе качества трендового индикатора и осциллятора. '
                                       'С его помощью можно определить дальнейшее направление цены, потенциальную силу ценового движения, а также точки возможного разворота тренда.')
                        moving_avarage(ticker, start, end)

                st.markdown("---")
                checkbox_machine_learning = st.checkbox('Отобразить прогнозируемую, при помощи машинного обучения, цену', key = 3)
                if checkbox_machine_learning:
                    with st.expander('Описание:'):
                        st.caption('Прогноз цены основан на прошлых значениях акций. Прогнозирование выполняется с использованием модели линейной регрессии на основе прошлых данных. '
                            'Линейная регрессия - это простой метод, который довольно легко интерпретировать, но у него есть несколько очевидных недостатков. '
                            'Одна из проблем при использовании алгоритмов регрессии заключается в том, что модель перестраивается под столбец даты и месяца. '
                            'Вместо того, чтобы учитывать предыдущие значения с точки зрения прогнозирования, модель будет учитывать значение с той же даты месяц назад или с той же даты/месяца год назад.')
                    Machine_learning(ticker, today)
                st.markdown("---")
            except IndexError:
                st.warning('Повторите запрос')
            checkbox_get_fundamental_data = st.checkbox('Отобразить фундаментальный анализ', key = 4)
            try:
                if checkbox_get_fundamental_data:
                    get_fundamental_data(ticker)
                    with st.expander('Описание:'):
                        st.caption('Выбраны основные показатели, благодаря которым, можно определить дальнейшую перспективу компании.')
                st.markdown("---")
                checkbox_analysis = st.checkbox('Отобразить прогнозы аналитиков', key = 5)
                if checkbox_analysis:
                    with st.expander('Описание:'):
                        st.caption('Прогнозы аналитиков за 6 месяцев')
                    analysis(ticker, today)
                st.markdown("---")
            except TypeError:
                st.warning('Данные по акции не найдены')

        except KeyError:
            st.warning('Введите корректное название')
    else:
        st.error('Введите верный временной диапозон!')



if __name__ == '__main__':
    main()

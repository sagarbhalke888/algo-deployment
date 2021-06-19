from flask import Flask, render_template, request
from ta.trend import DPOIndicator
from ta.volume import VolumeWeightedAveragePrice
from tapy import Indicators
from talib.abstract import *
import pandas as pd
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas_datareader as web
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import plotly
import talib

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/adx")
def adx():
    return render_template("adx.html")

def vwap(dataframe, label='vwap', length=3, fillna=True):
        dataframe[label] = VolumeWeightedAveragePrice(high=dataframe['High'], low=dataframe['Low'], close=dataframe["Close"], volume=dataframe['Volume'], window=length, fillna=fillna).volume_weighted_average_price()
        return dataframe


@app.route('/callbackvwap', methods=['POST', 'GET'])
def callbackvwap():
    symbol = request.form.get("fname")
    source = request.form.get("ohlc")
    length = request.form.get("len")
    return vwapCALCULATION(symbol,length)


@app.route('/callbackadx', methods=['POST', 'GET'])
def callbackadx():
    symbol = request.form.get("fname")
    length = request.form.get("len")
    return adxCALCULATION(symbol,length)


def adxCALCULATION(symbol,length):
    length = int(length)
    start = dt.datetime(2021,1,1)
    end = dt.datetime.now()
    data = web.DataReader(symbol,'yahoo', start, end).reset_index()
    ADX1 = ADX(data["High"],data["Low"],data["Close"],timeperiod=length)
    dates = data['Date']
    fig2 = go.Figure(go.Scatter(x=dates,y=ADX1))
    fig2.update_layout(
    title={
        'text': symbol,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

    graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON



def vwapCALCULATION(symbol,length):

    start = dt.datetime(2021,1,1)
    end = dt.datetime.now()
    data = web.DataReader(symbol,'yahoo', start, end).reset_index()

    finalvwap = vwap(data,length)
    dates = data['Date']

    fig = go.Figure(data=[go.Candlestick(x=dates,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'])])

    fig.add_trace(go.Scatter(x=dates,y=finalvwap['vwap']))


@app.route('/callbackalligator', methods=['POST', 'GET'])
def callbackalligator():
    symbol = request.form.get("fname")
    period_jaws = int(request.form.get("period_jaws"))
    period_teeth = int(request.form.get("period_teeth"))
    period_lips = int(request.form.get("period_lips"))
    shift_jaws = int(request.form.get("shift_jaws"))
    shift_teeth = int(request.form.get("shift_teeth"))
    shift_lips = int(request.form.get("shift_lips"))


    return alligatorCALCULATION(symbol,period_jaws,period_teeth,period_lips,shift_jaws,shift_teeth,shift_lips)

@app.route('/callbackha', methods=['POST', 'GET'])
def callbackha():
    print(request.args.get('data'))
    return haCALCULATION(request.args.get('data'))

@app.route('/callbackcci', methods=['POST', 'GET'])
def callbackcci():
    
    symbol = request.form.get("fname")
    source = request.form.get("ohlc")
    length = request.form.get("len")
    return CCICALCULATION(symbol,source,length)


@app.route('/callback', methods=['POST', 'GET'])
def callback():

    symbol = request.form.get("fname")
    source = request.form.get("ohlc")
    length = request.form.get("len")

    return BOLLINGERBANDCALC(symbol,source,length)

@app.route('/callbackdpo', methods=['POST', 'GET'])
def callbackdpo():

    symbol = request.form.get("fname")
    length = request.form.get("len")

    return dpoCALCULATION(symbol,length)

def dpoCALCULATION(symbol,length):
    length = int(length)
    start = dt.datetime(2021,1,1)
    end = dt.datetime.now()
    data = web.DataReader(symbol,'yahoo', start, end).reset_index()

    DPOIndicator_out =DPOIndicator(data["Close"], window = length)
    dpo = DPOIndicator_out.dpo()
    dates = data['Date']

    fig2 = go.Figure(go.Scatter(x=dates,y=dpo))
    fig2.update_layout(
    title={
        'text': symbol,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

    graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


@app.route('/callbacksma', methods=['POST', 'GET'])
def callbacksma():
    symbol = request.form.get("fname")
    source = request.form.get("ohlc")
    length = int(request.form.get("len"))
    return smaCALCULATION(symbol,source,length)

@app.route('/callbackwma', methods=['POST', 'GET'])
def callbackwma():
    symbol = request.form.get("fname")
    source = request.form.get("ohlc")
    length = int(request.form.get("len"))
    return wmaCALCULATION(symbol,source,length)

@app.route('/callbackema', methods=['POST', 'GET'])
def callbackema():
    symbol = request.form.get("fname")
    source = request.form.get("ohlc")
    length = request.form.get("len")
    return emaCALCULATION(symbol,source,length)

@app.route('/callbackmacd', methods=['POST', 'GET'])
def callbackmacd():
    symbol = request.form.get("fname")
    source = request.form.get("ohlc")
    lengthFast = request.form.get("lenfast")
    lengthSlow = request.form.get("lenslow")
    return macdCALCULATION(symbol,source,lengthFast,lengthSlow)

@app.route('/callbackrsi', methods=['POST', 'GET'])
def callbackrsi():
    symbol = request.form.get("fname")
    source = request.form.get("ohlc")
    length = int(request.form.get("len"))
    return rsiCALCULATION(symbol,source,length)

   
@app.route('/bb',methods=['POST', 'GET'])
def index():
    return render_template('bb.html')


@app.route('/cci')
def cci():
    return render_template('cci.html')

@app.route('/vwap')
def vwap():
    return render_template('vwap.html')

@app.route('/allig')
def allig():
    return render_template('allig.html')

def BOLLINGERBANDCALC(symbol,source,length):
    # int_features = [x for x in request.form.data()]
    # instrument = [np.array(int_features)]
    # instruments = request.form['symbol']

    print("***********************************************")
    # print(request.form)
    print("************************************************")
    start = dt.datetime(2021,1,1)
    end = dt.datetime.now()
    data = web.DataReader(symbol,'yahoo', start, end).reset_index()
    period = int(length)
    multiplier = 2
    data['MiddleBand'] = data[source].rolling(period).mean()
    data['UpperBand'] = data[source].rolling(period).mean() + data[source].rolling(period).std() * multiplier
    data['LowerBand'] = data[source].rolling(period).mean() - data[source].rolling(period).std() * multiplier

    dates = data['Date']

    fig = go.Figure(data=[go.Candlestick(x=dates,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'])])

    fig.add_trace(go.Scatter(x=dates,y=data['MiddleBand']))
    fig.add_trace(go.Scatter(x=dates,y=data['UpperBand']))
    fig.add_trace(go.Scatter(x=dates,y=data['LowerBand'] ))

    fig.update_layout(
    title={
        'text': symbol,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


def CCICALCULATION(symbol,source,length):
    start = dt.datetime(2021,1,1)
    end = dt.datetime.now()
    data = web.DataReader(symbol,'yahoo', start, end).reset_index()
    real = CCI(data['High'], data['Low'], data['Close'], timeperiod=20)
    dates = data['Date']
    fig2 = go.Figure(go.Scatter(x=dates,y=real))
    fig2.update_layout(
    title={
        'text': symbol,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON




# @app.route('/adx')
# def adx():
#     return render_template('adx.html')


# def adxCALCULATION(symbol):



#     fig2.update_layout(title={
#         'text': symbol,
#         'y':0.9,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'})
#     graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
#     return graphJSON

@app.route('/ha')
def ha():
    return render_template('ha.html')

@app.route('/dpo')
def dpo():
    return render_template('dpo.html')

def haCALCULATION(symbol):
    print("Inside HA")
    start = dt.datetime(2021,1,1)
    end = dt.datetime(2021,6,1)
    data = web.DataReader(symbol,'yahoo', start, end).reset_index()
    HAdf = data[['Open', 'High', 'Low', 'Close']]

    HAdf['CLOSE'] = round(((data['Open'] + data['High'] + data['Low'] + data['Close'])/4),2)

    for i in range(len(data)):
        if i == 0:
            HAdf.iat[0,0] = round(((data['Open'].iloc[0] + data['Close'].iloc[0])/2),2)
        else:
            HAdf.iat[i,0] = round(((HAdf.iat[i-1,0] + HAdf.iat[i-1,3])/2),2)
    HAdf['High'] = HAdf.loc[:,['Open', 'Close']].join(data['High']).max(axis=1)
    HAdf['Low'] = HAdf.loc[:,['Open', 'Close']].join(data['Low']).min(axis=1)
    fig2 = go.Figure(data=[go.Candlestick(x=data['Date'],
                    open=HAdf.Open,
                    high=HAdf.High,
                    low=HAdf.Low,
                    close=HAdf.Close)])

    fig2.update_layout(title={
        'text': symbol,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/macd')
def macd():
    return render_template('macd.html')


def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2)
    macd.columns = ["macd"]

    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean())
    signal.columns = ["signal"]
    
    hist = pd.DataFrame(macd['macd'] - signal['signal'])
    hist.columns = ["hist"]
    print(hist.columns)
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df


def macdCALCULATION(symbol,source,lengthFast,lengthSlow):
    lengthFast = int(lengthFast)
    lengthSlow = int(lengthSlow)
    start = dt.datetime(2021,1,1)
    end = dt.datetime.now()
    data = web.DataReader(symbol,'yahoo', start, end).reset_index()
    close = data[source]
    macd = get_macd(close, lengthFast, lengthSlow, 9)
    dates = data['Date']

    fig2 = go.Figure(data=[go.Bar(x=dates,
                        y = macd['hist'])])

    fig2.add_trace(go.Scatter(x=dates,y=macd['signal']))
    fig2.add_trace(go.Scatter(x=dates,y=macd['macd']))



    fig2.update_layout(title={
        'text': symbol,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route('/sma')
def sma():
    return render_template('sma.html')


def smaCALCULATION(symbol,source,length):
    start = dt.datetime(2021,1,1)
    end = dt.datetime.now()
    data = web.DataReader(symbol,'yahoo', start, end).reset_index()
    simpleMovingAverage = SMA(data, timeperiod=length, price=source)
    dates = data['Date']
    fig2 = go.Figure(data=[go.Candlestick(x=dates,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'])])
    fig2.add_trace(go.Scatter(x=dates,y=simpleMovingAverage))

    fig2.update_layout(title={
        'text': symbol,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route('/ema')
def ema():
    return render_template('ema.html')

def emaCALCULATION(symbol,source,length):
    start = dt.datetime(2021,1,1)
    end = dt.datetime.now()
    length = int(length)
    data = web.DataReader("GOOG",'yahoo', start, end).reset_index()
    ExponentialMovingAverage = EMA(data, timeperiod=length, price=source)
    dates = data['Date']
    fig2 = go.Figure(data=[go.Candlestick(x=dates,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'])])
    fig2.add_trace(go.Scatter(x=dates,y=ExponentialMovingAverage))



    fig2.update_layout(title={
        'text': symbol,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route('/wma')
def wma():
    return render_template('wma.html')

def wmaCALCULATION(symbol,source,length):
    start = dt.datetime(2021,1,1)
    end = dt.datetime.now()
    data = web.DataReader(symbol,'yahoo', start, end).reset_index()

    WeightedMovingAverage = WMA(data, timeperiod=length, price=source)
    dates = data['Date']
    fig2 = go.Figure(data=[go.Candlestick(x=dates,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'])])
    fig2.add_trace(go.Scatter(x=dates,y=WeightedMovingAverage))
    fig2.update_layout(title={
        'text': symbol,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route('/rsi')
def rsi():
    return render_template('rsi.html')

def rsiCALCULATION(symbol,source,length):
    start = dt.datetime(2021,1,1)
    end = dt.datetime.now()
    data = web.DataReader(symbol,'yahoo', start, end).reset_index()
    source = data[source]
    real = talib.RSI(source, timeperiod=length)
    dates = data['Date']
    fig2 = go.Figure(go.Scatter(x=dates,y=real))
    fig2.update_layout(title={
        'text': symbol,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def alligatorCALCULATION(symbol,period_jaws,period_teeth,period_lips,shift_jaws,shift_teeth,shift_lips):
    start = dt.datetime(2021,1,1)
    end = dt.datetime.now()
    data = web.DataReader(symbol,'yahoo', start, end).reset_index()
    i= Indicators(data)
    i.alligator(period_jaws=period_jaws,
                     period_teeth=period_teeth,
                     period_lips=period_lips,
                     shift_jaws=shift_jaws,
                     shift_teeth=shift_teeth,
                     shift_lips=shift_lips,
                     column_name_jaws='alligator_jaw',
                     column_name_teeth='alligator_teeth',
                     column_name_lips='alligator_lips')
    data = i.df
    
    dates = data['Date']

    fig = go.Figure(data=[go.Candlestick(x=dates,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'])])

    fig.add_trace(go.Scatter(x=dates,y=data['alligator_jaw']))
    fig.add_trace(go.Scatter(x=dates,y=data['alligator_teeth']))
    fig.add_trace(go.Scatter(x=dates,y=data['alligator_lips'] ))
    fig.update_layout(title={
        'text': "symbol",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON








@app.route("/pattern")
def pattern():
    return render_template("pattern.html") 






app.run(host="0.0.0.0",debug=True)


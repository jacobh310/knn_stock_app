import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime
import util

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color:#295E61 ;'>Stock Price Direction Prediction</h1>",
            unsafe_allow_html=True)

st.title("Stock Price Direction Prediction")

st.write("""
This Web App uses the KNearest Neighbor Repressor Algorithm from Sklearn to predict the direction of the chosen 
security. 
Every day this web app will output either a buy or pass signal on for the next trading day. A buy signal is generated 
when 
the model predicts that the next trading day's close will be higher than the most recent. The model was trained on the 
SPY etf but you can see how it performs vs other stocks **This model is for educational and demonstration purposes 
only. This is in no way financial advise**
""")

col1, col2 = st.beta_columns(2)

stocks = ("SPY", "AAPL", "GOOG", "MSFT", "TSLA", "AMD")
ticker = col1.selectbox("Select which Stock to Predict", stocks)
start_date = '2000-01-01'
ticker_futures = 'ES=F'
ticker_vix = '^VIX'

stock = yf.Ticker(ticker).history(period='1d', start=start_date)
stock = stock.drop(columns=['Dividends', 'Stock Splits'])

df = util.final_df(ticker, ticker_vix, ticker_futures, start_date, 3, 1)
today = df[-1:]
df = df.dropna()

model = KNeighborsRegressor(n_neighbors=9)
columns = ['50MA_2', '150MA_1', '10/50STD_3', '50/150MA_2', 'Futures_Close_1', 'Vix_Close/20MA_3', '10_50STD_3']
label = 'Target_pct_change'

tom_prediction = util.give_prediction(df, columns, label, model, today)[0][0] * 100

y_hat, _ = util.split_scale_train_yhat(df, columns, label, model)
final_eval = util.hold_eval(df, y_hat)
prediction = y_hat.apply(lambda x: 1 if x > 0 else 0)
prices = final_eval['Close']
total = 10000
start = final_eval.reset_index()['Date'][0].date()
precision = precision_score(final_eval['Target'], prediction)
benchmark = df['Target'].value_counts(normalize=True)[1]

stock = stock.reset_index()
stock['15MA'] = stock['Close'].rolling(window=15).mean()
stock['50MA'] = stock['Close'].rolling(window=50).mean()
stock['150MA'] = stock['Close'].rolling(window=150).mean()
stock = stock.dropna()

col1.write(" ")
if tom_prediction > 0:
    col1.write("""## Todays' Signal: BUY""", color="green")
elif tom_prediction <= 0:
    col1.write("""## Todays' Signal: PASS""", color="red")

col1.write(f'The model performed with a precision of {precision * 100:0.2f}% on the test data from {start} until today')
# col1.write("---------------------------------------------------------------")
col1.write(f"""Below we simulated and evaluated the model's signals with trades starting on {start} through today.  
            The model will start with ${total} and only trade {ticker}. We will also show the returns of just 
            investing  
            ${total} into {ticker} in the same time period""")
col1.write("### Chose a reward to risk ratio")
rr = col1.slider(" ", 1, 10, 6)
p_l = util.pl(0, total, prices, rr, prediction)
stock_gain = (final_eval['Close'][-1] / final_eval['Close'][0]) * 10000


def plot_pl( model_gain, inv_gain):
    x = ['Model Trading', 'Invested']
    y = [round(model_gain,2), round(inv_gain,2)]
    trace = (go.Bar(x=x,y=y, text=y, textposition='auto'))
    layout = go.Layout(title= 'Returns',
                       yaxis_title='Gain ($)',
                       margin=go.layout.Margin(b=0),
                       )


    col1.plotly_chart(go.Figure(data=trace, layout=layout))

plot_pl(p_l-total, stock_gain-total)


col1.write(f"""The model trading column shows the return the model's signals and the invest columns shows the return 
from investing ${total} in {ticker}. **Again this is not financial advice and is only for demonstration and 
educational purposes**""")


def plot_candle_sticks(days):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(x=stock['Date'][-days:], open=df['Open'][-days:], high=df['High'][-days:], low=df['Low'][-days:],
                       close=stock['Close'][-days:], name='Candle stick'))
    fig.add_trace(go.Scatter(x=stock['Date'][-days:], y=stock['15MA'][-days:], line=dict(color='purple', width=1),
                             name='10 Day Moving Average'))
    fig.add_trace(go.Scatter(x=stock['Date'][-days:], y=stock['50MA'][-days:], line=dict(color='orange', width=1),
                             name='50 Day Moving Average'))
    fig.add_trace(go.Scatter(x=stock['Date'][-days:], y=stock['150MA'][-days:], line=dict(color='blue', width=1),
                             name='150 Day Moving Average'))

    fig.layout.update(title=f'{ticker} Stock Price Chart for the past {days} days', yaxis_title="Stock Price ($)",
                      width=920, height=650)

    col2.plotly_chart(fig)


days = 450
plot_candle_sticks(days)

stock = stock.set_index('Date')
stock.index = stock.index.date
stock.reset_index(inplace=True)
stock = stock.rename(columns={'index': 'Date'})
last_5 = stock[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']].tail().round(2)

col2.write("""#### Data for the last 5 trading days""")
col2.dataframe(last_5)

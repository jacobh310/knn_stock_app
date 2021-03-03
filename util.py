import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler


ticker = 'SPY'
ticker_futures = 'ES=F'
ticker_vix = '^VIX'
start_date = '2000-01-01'

def rsi(df, period):
    prices = df.Close
    # upward list will contain the the dollar amount a stock when up from the previous days close if it went up
    upward = []
    for i in range(len(prices)):
        if i == 0:
            upward.append(np.nan)
        elif prices[i] > prices[i - 1]:
            upward.append(prices[i] - prices[i - 1])
        else:
            upward.append(0)

    # the downward list will contain the dollar amount a stock when down from
    downward = []
    for i in range(len(prices)):
        if i == 0:
            downward.append(np.nan)
        elif prices[i] < prices[i - 1]:
            downward.append(prices[i - 1] - prices[i])
        else:
            downward.append(0)

    # making a datafram to help oraginze the data
    rsi = pd.DataFrame(
        {'Date': df.reset_index()['Date'], 'Close': df.reset_index()['Close'], 'upward': upward, 'downward': downward})


    upward_avg = []
    upward = rsi['upward']
    for i in range(len(upward)):
        period_avg = upward[i - period:i].mean()
        if i < period:
            upward_avg.append(np.nan)
        elif i == period:
            upward_avg.append(period_avg)
        else:
            upward_avg.append((upward_avg[-1] * (period - 1) + upward[i]) / period)

    # downward average (downward_avg) is calculated the same way as upward except I used the downward column
    downward_avg = []
    downward = rsi['downward']
    for i in range(len(downward)):
        period_avg = downward[i - period:i].mean()
        if i < period:
            downward_avg.append(np.nan)
        elif i == period:
            downward_avg.append(period_avg)
        else:
            downward_avg.append((downward_avg[-1] * (period - 1) + downward[i]) / period)

    rsi['upward_avg'] = upward_avg
    rsi['downward_avg'] = downward_avg
    rsi = rsi.dropna()
    rsi['relative_strength'] = rsi.upward_avg / rsi.downward_avg
    rsi['rsi'] = 100 - (100 / (rsi['relative_strength'] + 1))

    df = df.iloc[period:]  # removing the rows that rsi needed to calculate
    df['rsi'] = rsi['rsi'].tolist()

    return df


def final_df(ticker, vix, futures, start_date, m, n):
    stock = yf.Ticker(ticker).history(period='1d', start=start_date)
    stock = stock.drop(columns=['Dividends', 'Stock Splits'])

    vix = yf.Ticker(vix).history(period='1d', start=start_date)
    vix = vix.drop(columns=['Volume', 'Low', 'High', 'Open', 'Dividends', 'Stock Splits'])
    vix['20MA'] = vix['Close'].rolling(window=20).mean()
    vix['Close/20MA'] = vix['Close'] / vix['20MA']
    vix.columns = ['Vix_' + i for i in vix.columns]

    futures = yf.Ticker(futures).history(period='1d', start=start_date)
    futures = futures.dropna()
    future = futures.drop(columns=['Low', 'High', 'Open', 'Volume', 'Dividends', 'Stock Splits'])
    futures['20MA'] = futures['Close'].rolling(window=20).mean()
    futures['Close/20MA'] = futures['Close'] / futures['20MA']
    futures.columns = ['Futures_' + i for i in futures.columns]

    df = pd.concat([futures, vix, stock], axis=1, join='inner')
    df = df.dropna()

    df['pct_change'] = df['Close'].pct_change()
    df['50_Volume'] = df['Volume'].rolling(window=30).mean()
    df['10MA'] = df['Close'].rolling(window=10).mean()
    df['50MA'] = df['Close'].rolling(window=50).mean()
    df['150MA'] = df['Close'].rolling(window=150).mean()
    df['50STD'] = df['Close'].rolling(window=50).std() * 2
    df['10STD'] = df['Close'].rolling(window=10).std() * 2

    df = df.dropna(axis=0)

    df = rsi(df, 14)

    df['Close/10MA'] = df['Close'] / df['10MA']
    df['10/50MA'] = df['10MA'] / df['50MA']
    df['50/150MA'] = df['50MA'] / df['150MA']
    df['10/50STD'] = df['10STD'] / df['50STD']
    df['Vol/30Vol'] = df['Volume'] / df['50_Volume']

    df['Close-10MA'] = df['Close'] - df['10MA']
    df['10-50MA'] = df['10MA'] - df['50MA']
    df['50-150MA'] = df['50MA'] - df['150MA']
    df['10-50STD'] = df['10STD'] - df['50STD']

    df['range_hl'] = df['High'] - df['Low']
    df['range_oc'] = df['Close'] - df['Open']

    df['Close_10MA'] = df['Close-10MA'].apply(lambda x: 1 if x > 0 else 0)
    df['10_50MA'] = df['10-50MA'].apply(lambda x: 1 if x > 0 else 0)
    df['50_150MA'] = df['50-150MA'].apply(lambda x: 1 if x > 0 else 0)
    df['10_50STD'] = df['10-50STD'].apply(lambda x: 1 if x > 0 else 0)

    df = df.reset_index()
    df['Daily_pct_change'] = df['Close'].pct_change()
    df['Target_Close'] = df['Close'].shift(-n)
    df['Target_pct_change'] = df['Close'].pct_change(n).shift(-n)
    df['Target'] = df['Target_pct_change'].apply(lambda x: 1 if x >= 0 else (0 if x < 0 else np.nan))

    cols = df.columns.drop(['Date', 'Target_Close', 'Target_pct_change', 'Target'])
    for i in range(1, m + 1):
        for col in cols:
            df[col + '_' + str(i)] = df[col].shift(i)

    return df

def give_prediction(df, columns, label, model,today):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    y = np.expand_dims(df[label], -1)

    x = x_scaler.fit_transform(df[columns])
    y = y_scaled = y_scaler.fit_transform(y)

    model.fit(x, y)
    prediction = model.predict(today[columns])
    prediction = y_scaler.inverse_transform(prediction)

    return prediction


def split_scale_df(df, columns, label):
    cutoff = int(df.shape[0] * .8)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    y = np.expand_dims(df[label], -1)

    x_scaled = x_scaler.fit_transform(df[columns])
    y_scaled = y_scaler.fit_transform(y)

    X_train_scaled = x_scaled[:cutoff]
    y_train_scaled = y_scaled[:cutoff]

    X_test_scaled = x_scaled[cutoff:]
    y_test_scaled = y_scaled[cutoff:]

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler


def split_scale_train_yhat(df, features, label, model):
    X_train, y_train, X_test, y_test, y_scaler = split_scale_df(df, features, label)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    y_hat = y_scaler.inverse_transform(y_hat.reshape(-1, 1))
    y_test = y_scaler.inverse_transform(y_test)
    y_test = pd.Series(y_test[:, 0])
    y_hat = pd.Series(y_hat[:, 0])

    return y_hat, y_test

def pl(curr,total,prices,rr,prediction):
    risk = 0.02
    tp = risk*rr
    stop = len(prices)
    if curr == stop:
        return total

    price = prices[curr]
    risk_amount = risk * total
    shares = total // price
    tp_price = price + price * tp
    reward_amount = shares * tp_price - total
    stop_price = price - price * risk


    if prediction[curr] == 1:

        for j, price_j in enumerate(prices[curr:]):

            if price_j <= stop_price:
                total = total - risk_amount
                curr = curr + j
                return pl(curr, total, prices,rr, prediction)

            if price_j >= tp_price:
                total = total + reward_amount
                curr = curr + j
                return pl(curr, total, prices,rr, prediction)
            if curr + j == stop -1:
                change = price_j - price
                total = total + change*shares
                curr = stop
                return pl(curr, total,prices,rr, prediction)

    else:
        curr = curr + 1
        return pl(curr,total,prices,rr,prediction)


def split_train_pl(df, features, label, model):
    total = 1000

    y_hat, _ = split_scale_train_yhat(df, features, label, model)
    prediction = y_hat.apply(lambda x: 1 if x > 0 else 0)
    cutoff = int(0.8 * df.shape[0])
    prices = df['Close'][cutoff:]

    p_l = pl(0, total, prices, 6, prediction)

    return p_l


def hold_eval(df, y_hat):
    cutoff = int(df.shape[0] * .8)
    evaluation = df.copy()

    evaluation = df[['Date', 'Close', 'Target_Close', 'Target_pct_change', 'Target']].iloc[cutoff:]
    evaluation.set_index('Date', inplace=True)
    evaluation['Predicted_pct_change'] = y_hat.values
    evaluation['Predicted_Close'] = evaluation['Close'] * evaluation['Predicted_pct_change'] + evaluation['Close']
    evaluation['Predcited_direction'] = evaluation['Predicted_pct_change'].apply(lambda x: 1 if x > 0 else 0)

    return evaluation

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

def fetch_data(tickers, start_date, end_date, interval='1mo'):
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval)
    return data

# 동적 포트폴리오 전략: 1개월 마다 주가변화율(pct_change)이 낮은 주식을 빼고 주가변화율이 더 높은 주식으로 교체하는 전략 
def pflio(df, m, x):
    portfolio = []
    monthly_ret = [0]

    for i in range(len(df)):
        if portfolio:
            monthly_ret.append(df[portfolio].iloc[i, :].mean())
            bad_stocks = df[portfolio].iloc[i, :].nsmallest(x).index.tolist()
            portfolio = [t for t in portfolio if t not in bad_stocks]
        fill = m - len(portfolio)
        new_picks = [t for t in df.iloc[i, :].nlargest(fill).index if t not in portfolio]
        portfolio.extend(new_picks)
    
    monthly_ret_df = pd.DataFrame(np.array(monthly_ret), columns=['mon_ret'])
    return monthly_ret_df

def CAGR(df):
    df["cum_return"] = (1 + df["mon_ret"]).cumprod()
    n = len(df) / 12
    cagr = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return cagr

def volatility(df, period=12):
    vol = df["mon_ret"].std() * np.sqrt(period)
    return vol

def sharpe(df, rf):
    sr = (CAGR(df) - rf) / volatility(df)
    return sr

# max drawdown
def max_dd(df):
    df["cum_return"] = (1 + df["mon_ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"] / df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd

# pflio 전략의 CAGR, Sharpe, Max Drawdown을 테스팅: rolling window analysis
def rolling_backtest(df, window_years=3, step_months=12, m=6, x=3, rf=0.043):
    results = []
    window_size = window_years * 12
    step_size = step_months
    for start in range(0, len(df)-window_size + 1, step_size):
        rolling_df = df.iloc[start: start + window_size]
        portfolio_returns = pflio(rolling_df, m, x)
        cagr = CAGR(portfolio_returns)
        sharpe_ratio = sharpe(portfolio_returns, rf)
        max_drawdown = max_dd(portfolio_returns)

        results.append({
            'start_date': df.index[start],
            'end_date': df.index[start + window_size -1],
            'CAGR': cagr, 
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        })

    return pd.DataFrame(results)

def plot_metrics(results):
    # Visualizing the results
    fig, ax = plt.subplots(figsize=(12, 6))
    results.plot(x='start_date', y=['CAGR', 'Sharpe Ratio', 'Max Drawdown'], kind='line', ax=ax)
    plt.title('Rolling Backtest Metrics')
    plt.xlabel('Start Date')
    plt.ylabel('Metrics')
    plt.grid()
    plt.legend(['CAGR', 'Sharpe Ratio', 'Max Drawdown'])
    plt.show()

if __name__ == '__main__':

    # tickers = ["MSFT", "IBM", "HD", "GS", "XOM", "DIS", "KO", "CSCO", "CVX", "CAT", "BA", "AAPL", "AXP", "MMM"]
    tickers = ["IBM", "HD",  "KO", "CSCO", "CVX", "CAT", "BA", "AXP", "MMM"]
    
    start = (dt.datetime.today() - dt.timedelta(days=365 * 20)).strftime('%Y-%m-%d')
    end = dt.datetime.today().strftime('%Y-%m-%d')

    df_ohlcv = fetch_data(tickers, start, end, interval='1mo')
    df_cls = df_ohlcv['Adj Close'].dropna(how='all')
    df_cls_ret = df_cls.pct_change(fill_method=None).fillna(0)

    # Rolling backtest with a 5-year window and 1-year step
    results = rolling_backtest(df_cls_ret)

    # Display the rolling backtest results
    print(results)

    # Display the metics
    plot_metrics(results)
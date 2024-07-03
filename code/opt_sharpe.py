import pandas as pd
import yfinance as yf
import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.optimize as sco

# Define Yahoo Finance stock downloading function 
def get_data_by_day(ticker_list, start, end):
    interval = '1d'
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker']
    data_list = []
    for ticker in ticker_list:
        df = yf.download(ticker, start=start, end=end, interval=interval)
        if not df.empty:
            df['ticker'] = ticker
            data_list.append(df)
    download_data = pd.concat(data_list).reset_index()
    download_data = download_data[columns]  # Reorder and select relevant columns
    
    unique_dates = download_data['Date'].drop_duplicates().sort_values(ascending=False).reset_index(drop=True)
    date_map = pd.Series(range(1, len(unique_dates) + 1), index=unique_dates)
    download_data['dayseq'] = download_data['Date'].map(date_map)
    download_data = download_data.sort_values(['ticker', 'Date'], ascending=[False, False])
    return download_data

# Example usage
sym = pd.read_csv('smb.csv', encoding='latin1')
tks = list(sym.Symbol)
day_data = get_data_by_day(tks, '2021-06-30', '2024-06-30')
day_data.to_csv('smb_price.csv', index=False)
tks = list(day_data.ticker)

# Remove some tickers with fewer trading days
D =  day_data['ticker'].value_counts().reset_index()
rem = list(D[D['count'] < 754]['ticker'])
day_data = day_data[day_data.ticker.isin(rem) == False]
tks = list(day_data.ticker)


# Define a function to compute the Sharpe ratio for a single stock 
def sharpe(tk, day_data, risk_free_rate=0.0005):
    data = day_data[day_data['ticker'] == tk]
    
    # Ensure the data is sorted by date
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    
    # Calculate daily returns
    data['Daily Return'] = data['Close'].pct_change()
        
    # Calculate the average daily return
    avg_daily_return = data['Daily Return'].mean()
    
    # Calculate the standard deviation of daily returns
    std_daily_return = data['Daily Return'].std()
    
    # Calculate the Sharpe ratio (annualized)
    sharpe_ratio = (avg_daily_return - risk_free_rate) / std_daily_return
    annual_sharpe_ratio = sharpe_ratio * np.sqrt(252)  # Assuming 252 trading days in a year
    
    return annual_sharpe_ratio

# Put all Sharpe ratios of the selected stocks into a Data Frame
sp_lst = []
for tk in set(tks):
    sp = sharpe(tk, day_data, risk_free_rate=0.0005)
    sp_tk = [tk, sp]
    sp_lst.append(sp_tk)

sp_df = pd.DataFrame(sp_lst, columns=['ticker', 'sp'])
sp_df = sp_df.sort_values(['sp'], ascending=False)
print(sp_df.head(20))


result_df = pd.DataFrame()

for ticker in day_data['ticker'].unique():
    df_ticker = day_data[day_data['ticker'] == ticker].sort_values(by='Date')
    x_vals = np.arange(len(df_ticker))
    spline_price = make_interp_spline(x_vals, df_ticker['Close'], k=3)(np.linspace(x_vals.min(), x_vals.max(), 300))
    spline_volume = make_interp_spline(x_vals, df_ticker['Volume'], k=3)(np.linspace(x_vals.min(), x_vals.max(), 300))

    # Select 30 evenly spaced points from the smoothed curves
    indices = np.linspace(0, 299, 30, dtype=int)
    # Prepare the smoothed data DataFrame for this ticker
    df_smooth = pd.DataFrame({
        'ticker': ticker,
        'smooth_price': spline_price[indices],
        'smooth_volume': spline_volume[indices],
        'seq': np.arange(30, 0, -1)})
    # Append the smoothed data
    result_df = pd.concat([result_df, df_smooth], ignore_index=True)
    

# Data pivoting
result_df_pivoted = result_df.pivot(index='ticker', columns='seq', values=['smooth_price', 'smooth_volume'])
result_df_pivoted.columns = [f'{val}_{i}' for val, i in result_df_pivoted.columns]
result_df_pivoted.reset_index(inplace=True)

# Data standardization
scaler = StandardScaler()
price_columns = [col for col in result_df_pivoted.columns if 'smooth_price' in col]
result_df_pivoted[price_columns] = scaler.fit_transform(result_df_pivoted[price_columns])
result_df_std = result_df_pivoted[['ticker'] + price_columns]    

K = range(1, 10)
sum_of_squared_distances = []
for k in K:
    kmeans = KMeans(n_clusters=k).fit(result_df_std.drop('ticker', axis=1))
    sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('K values')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal K')
plt.show()


# Optimal number of clusters 4 
optimal_k = 4  # Adjust based on elbow method results
kmeans = KMeans(n_clusters=optimal_k)
result_df_std['cluster'] = kmeans.fit_predict(result_df_std.drop('ticker', axis=1))

# Performance evaluation
silhouette_score = metrics.silhouette_score(result_df_std.drop(['ticker', 'cluster'], axis=1), result_df_std['cluster'])
print(f'Silhouette Score: {silhouette_score}')
clus_df = result_df_std[['ticker', 'cluster']]

# Merge stock cluster data with the Sharpe ratio data
sp_df = pd.merge(sp_df, clus_df, on='ticker', how='inner')
sp_df['cluster'] = sp_df['cluster'] + 1
sp_df = sp_df.sort_values(['cluster', 'sp'], ascending=[1, 0])

# Choose higher Sharpe ratio stocks from each cluster. we get 9 stocks

c1 = ['SMCI', 'MCK', 'VRTX', 'PANW']
c2 = ['CNSWF']
c3 = ['LLY', 'AVGO']
c4 = ['NVDA', 'NVO']

tickers = c1 + c2 + c3 + c4

# Filter data for selected stocks
data = day_data[day_data.ticker.isin(tickers)]
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Calculate daily returns for each stock
data['Daily Return'] = data.groupby('ticker')['Close'].pct_change()

# Pivot the data to have dates as index and tickers as columns
pivot_data = data.pivot_table(index='Date', columns='ticker', values='Daily Return')

# Calculate mean returns and covariance matrix
mean_returns = pivot_data.mean()
cov_matrix = pivot_data.cov()

# Assume the risk-free rate is 0.05% per day
risk_free_rate = 0.0005

# Function to calculate portfolio statistics
def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio


# Function to minimize (negative Sharpe ratio)
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    return -portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[2]

# Constraints and bounds
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for _ in range(len(tickers)))

# Initial guess (equal weights)
initial_guess = [1/len(tickers) for _ in range(len(tickers))]

# Optimize
optimized_results = sco.minimize(negative_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix, risk_free_rate), method='SLSQP', bounds=bounds, constraints=constraints)

# Get the optimized weights
optimized_weights = optimized_results.x

print("Optimized Weights:")
for ticker, weight in zip(tickers, optimized_weights):
    print(f"{ticker}: {weight:.4f}")

# Calculate the Sharpe ratio with the optimized weights
portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_stats(optimized_weights, mean_returns, cov_matrix, risk_free_rate)

print(f"Annualized Portfolio Return: {portfolio_return * 252:.4f}")
print(f"Annualized Portfolio Volatility: {portfolio_volatility * np.sqrt(252):.4f}")
print(f"Annualized Sharpe Ratio: {sharpe_ratio * np.sqrt(252):.4f}")
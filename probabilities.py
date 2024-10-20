import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

nvidia_data = pd.read_csv("data.csv")

nvidia_data['Date'] = pd.to_datetime(nvidia_data['Date'], format='%m/%d/%Y')
nvidia_data.sort_values('Date', inplace=True)

nvidia_data['50_MA'] = nvidia_data['Close'].rolling(window=50).mean()

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

nvidia_data['RSI'] = calculate_rsi(nvidia_data)
nvidia_data.dropna(subset=['50_MA', 'RSI'], inplace=True)

def calculate_conditional_probabilities(data, end_date, decay_factor):
    """
    Calculates conditional probabilities of buy signals and next-day movements
    using a skewed weight favoring more recent movements.

    Parameters:
    - data: DataFrame containing stock data with calculated indicators.
    - end_date: The date up to which the data will be used for training (inclusive).
    - decay_factor: The exponential decay factor controlling how fast weights decrease (default=0.01).

    Returns:
    - ma_buy_probabilities: Conditional probabilities for the 50-day MA buy signal.
    - rsi_buy_probabilities: Conditional probabilities for the RSI buy signal.
    - next_day_probabilities: Weighted probabilities for the next day's movement.
    - filtered_data: Data up to the specified end_date.
    """
    filtered_data = data[data['Date'] < pd.to_datetime(end_date)]

    filtered_data['MA_Buy'] = filtered_data['Close'] > filtered_data['50_MA']
    filtered_data['RSI_Buy'] = (filtered_data['RSI'].shift(1) < 30) & (filtered_data['RSI'] >= 30)

    filtered_data['Price_Movement'] = np.where(
        filtered_data['Close'] > filtered_data['Close'].shift(1), 'Increase',
        np.where(filtered_data['Close'] < filtered_data['Close'].shift(1), 'Decrease', 'No Change')
    )

    ma_buy_counts = filtered_data.groupby(['Price_Movement', 'MA_Buy']).size().unstack(fill_value=0)
    rsi_buy_counts = filtered_data.groupby(['Price_Movement', 'RSI_Buy']).size().unstack(fill_value=0)

    ma_buy_probabilities = ma_buy_counts.div(ma_buy_counts.sum(axis=1), axis=0)
    rsi_buy_probabilities = rsi_buy_counts.div(rsi_buy_counts.sum(axis=1), axis=0)

    filtered_data['Days_Since'] = (filtered_data['Date'].max() - filtered_data['Date']).dt.days
    filtered_data['Weight'] = np.exp(-decay_factor * filtered_data['Days_Since'])

    weighted_movement_counts = filtered_data.groupby('Price_Movement')['Weight'].sum()

    next_day_probabilities = weighted_movement_counts / weighted_movement_counts.sum()

    return ma_buy_probabilities, rsi_buy_probabilities, next_day_probabilities, filtered_data

def plot_indicators(data):
    """
    Plots the 50-day moving average and RSI indicators over the dataset.

    Parameters:
    - data: DataFrame containing stock data with the calculated 50-day MA and RSI.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    ax1.plot(data['Date'], data['50_MA'], label='50-Day MA', color='red')
    ax1.set_title('NVIDIA Stock Price and 50-Day Moving Average')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()

    ax2.plot(data['Date'], data['RSI'], label='RSI', color='green')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='red')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='red')
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Date')
    ax2.legend()

    plt.tight_layout()
    plt.show()

end_date_example = '2022-10-18'
ma_buy_probabilities_example, rsi_buy_probabilities_example, next_day_probabilities_example, filtered_data_example = \
    calculate_conditional_probabilities(nvidia_data, end_date_example, decay_factor=0.05)

print("50-Day MA Buy Signal Probabilities:")
print(ma_buy_probabilities_example)
print("\nRSI Buy Signal Probabilities:")
print(rsi_buy_probabilities_example)
print("\nNext Day Price Movement Probabilities with Skewed Weighting:")
print(next_day_probabilities_example)

plot_indicators(filtered_data_example)

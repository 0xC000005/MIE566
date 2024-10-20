import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

# Load the dataset
nvidia_data = pd.read_csv("data.csv")

# Convert the Date column to datetime format and sort the data by date
nvidia_data['Date'] = pd.to_datetime(nvidia_data['Date'], format='%m/%d/%Y')
nvidia_data.sort_values('Date', inplace=True)

# Calculate the 50-day moving average (MA)
nvidia_data['50_MA'] = nvidia_data['Close'].rolling(window=50).mean()

# Calculate the RSI (14-day period)
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

nvidia_data['RSI'] = calculate_rsi(nvidia_data)
nvidia_data.dropna(subset=['50_MA', 'RSI'], inplace=True)

def calculate_conditional_probabilities(data, end_date):
    """
    Calculates conditional probabilities of buy signals for MA and RSI given price movements
    using data up to a specified end_date, and estimates the probabilities of the next year's
    price movement using historical data.
    
    Parameters:
    - data: DataFrame containing stock data with calculated indicators.
    - end_date: The date up to which the data will be used for training (inclusive).
    
    Returns:
    - ma_buy_probabilities: DataFrame of conditional probabilities for the 50-day MA buy signal.
    - rsi_buy_probabilities: DataFrame of conditional probabilities for the RSI buy signal.
    - filtered_data: Filtered DataFrame used in the calculation.
    """
    # Filter data up to the specified end_date
    filtered_data = data[data['Date'] < pd.to_datetime(end_date)]
    
    # Define buy signals for the indicators
    filtered_data['MA_Buy'] = filtered_data['Close'] > filtered_data['50_MA']
    filtered_data['RSI_Buy'] = (filtered_data['RSI'].shift(1) < 30) & (filtered_data['RSI'] >= 30)
    
    # Classify price movements as 'Increase', 'Decrease', or 'No Change'
    filtered_data['Price_Movement'] = np.where(
        filtered_data['Close'] > filtered_data['Close'].shift(1), 'Increase',
        np.where(filtered_data['Close'] < filtered_data['Close'].shift(1), 'Decrease', 'No Change')
    )
    
    # Calculate counts for conditional probabilities of buy signals given price movements
    ma_buy_counts = filtered_data.groupby(['Price_Movement', 'MA_Buy']).size().unstack(fill_value=0)
    rsi_buy_counts = filtered_data.groupby(['Price_Movement', 'RSI_Buy']).size().unstack(fill_value=0)
    
    # Calculate probabilities
    ma_buy_probabilities = ma_buy_counts.div(ma_buy_counts.sum(axis=1), axis=0)
    rsi_buy_probabilities = rsi_buy_counts.div(rsi_buy_counts.sum(axis=1), axis=0)
    
    return ma_buy_probabilities, rsi_buy_probabilities, filtered_data

def calculate_backtesting_probabilities(data, end_date):
    """
    Calculates conditional probabilities of the price at time P_i going up or down 
    based on whether the price at time P_{i+1} went up or down using a backtesting approach.
    
    Parameters:
    - data: DataFrame containing stock data.
    - end_date: The date up to which the data will be used for training (inclusive).
    
    Returns:
    - conditional_probabilities: DataFrame showing the conditional probabilities 
      of P_i's movement given P_{i+1}'s movement.
    """
    # Filter data up to the specified end_date
    filtered_data = data[data['Date'] < pd.to_datetime(end_date)].copy()
    
    # Classify price movements as 'Increase' or 'Decrease' at P_i
    filtered_data['Price_Movement_i'] = np.where(
        filtered_data['Close'] > filtered_data['Close'].shift(1), 'Increase',
        np.where(filtered_data['Close'] < filtered_data['Close'].shift(1), 'Decrease', np.nan)
    )
    
    # Align P_i+1 to P_i's time frame by shifting Price_Movement_i forward
    filtered_data['Price_Movement_i+1'] = filtered_data['Price_Movement_i'].shift(-1)
    
    # Drop rows where either Price_Movement_i or Price_Movement_i+1 is NaN
    filtered_data.dropna(subset=['Price_Movement_i', 'Price_Movement_i+1'], inplace=True)
    
    # Calculate counts for conditional probabilities of P_i given P_i+1
    movement_counts = filtered_data.groupby(['Price_Movement_i+1', 'Price_Movement_i']).size().unstack(fill_value=0)
    
    # Calculate conditional probabilities P(P_i | P_i+1)
    conditional_probabilities = movement_counts.div(movement_counts.sum(axis=1), axis=0)
    
    return conditional_probabilities

def plot_indicators(data):
    """
    Plots the 50-day moving average and RSI indicators over the dataset.
    
    Parameters:
    - data: DataFrame containing stock data with the calculated 50-day MA and RSI.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 50-day MA and Close Price
    ax1.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    ax1.plot(data['Date'], data['50_MA'], label='50-Day MA', color='red')
    ax1.set_title('NVIDIA Stock Price and 50-Day Moving Average')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()

    # Plot RSI
    ax2.plot(data['Date'], data['RSI'], label='RSI', color='green')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='red')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='red')
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Date')
    ax2.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example: Calculate probabilities using data up to October 18, 2023
end_date_example = '2022-10-18'
ma_buy_probabilities_example, rsi_buy_probabilities_example, filtered_data_example = calculate_conditional_probabilities(nvidia_data, end_date_example)
next_year_probabilities_example = calculate_backtesting_probabilities(nvidia_data, end_date_example)

# Print the results
print("50-Day MA Buy Signal Probabilities:")
print(ma_buy_probabilities_example)
print("\nRSI Buy Signal Probabilities:")
print(rsi_buy_probabilities_example)
print("\nNext Year Price Movement Probabilities Based on Historical Data:")
print(next_year_probabilities_example)

# Plot the 50-day MA and RSI indicators
plot_indicators(filtered_data_example)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, lognorm, expon, gamma, alpha, beta, weibull_min

# Load the data from the attached file
file_path = 'data.csv'
data = pd.read_csv(file_path, parse_dates=['Date'])

# Calculate log returns
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

# Calculate the 50-day moving average of returns
data['50_MA'] = data['Log_Return'].rolling(window=50).mean()

# Calculate RSI based on returns
def calculate_rsi(returns, window=14):
    delta = returns.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data['Log_Return'])

# Filter out rows before the 60th day to have valid 50-day MA and RSI values
filtered_data = data.iloc[50:].dropna(subset=['50_MA', 'RSI'])

# Mapping for parameter labels per distribution
param_labels = {
    "Normal": ["mean", "std"],
    "Lognormal": ["shape", "location", "scale"],
    "Exponential": ["location", "scale"],
    "Gamma": ["shape", "location", "scale"],
    "Alpha": ["shape", "location", "scale"],
    "Beta": ["a", "b", "location", "scale"],
    "Weibull": ["c", "location", "scale"]
}

# Function to fit, plot distributions, and display equation with labeled parameters
def fit_and_plot(data, title, ax):
    ax.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data Histogram')

    # Fit different distributions and select the best one
    distributions = {
        "Normal": norm,
        "Lognormal": lognorm,
        "Exponential": expon,
        "Gamma": gamma,
        "Alpha": alpha,
        "Beta": beta,
        "Weibull": weibull_min
    }

    best_fit = None
    best_p_value = 0
    best_params = ()
    for name, distribution in distributions.items():
        # Fit distribution to data
        params = distribution.fit(data)
        
        # Generate PDF with fitted parameters
        x = np.linspace(min(data), max(data), 100)
        pdf_fitted = distribution.pdf(x, *params)
        ax.plot(x, pdf_fitted, label=f"{name} fit")

        # Goodness-of-fit test using a lambda function to apply the parameters to kstest
        d_stat, p_value = stats.kstest(data, lambda x: distribution.cdf(x, *params))
        
        if p_value > best_p_value:
            best_p_value = p_value
            best_fit = name
            best_params = params

    # Add details to plot
    ax.set_title(f"{title} - Best Fit: {best_fit} (p-value: {best_p_value:.4f})")
    ax.legend()

    # Create the labeled parameters text
    param_text = f"{best_fit} parameters:\n"
    labels = param_labels[best_fit]
    for label, value in zip(labels, best_params):
        param_text += f"{label}: {value:.4f}\n"

    # Display the labeled parameters on the plot
    ax.text(0.95, 0.6, param_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"))

    print(f"Best fit for {title}: {best_fit} distribution with p-value = {best_p_value:.4f}")
    print(f"Parameters: {dict(zip(labels, best_params))}")

# Set up subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Fit and plot distributions for each variable on a different subplot
fit_and_plot(filtered_data['Log_Return'].dropna(), 'Nvidia Log Returns', axs[0])
fit_and_plot(filtered_data['50_MA'].dropna(), '50-day Moving Average of Log Returns', axs[1])
fit_and_plot(filtered_data['RSI'].dropna(), 'RSI of Log Returns', axs[2])

plt.tight_layout()
plt.show()

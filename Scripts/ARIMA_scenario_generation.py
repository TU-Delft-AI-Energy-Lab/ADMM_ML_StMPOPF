"""
Stochastic Scenario Generation Using ARIMA


Description:
This script uses an ARIMA model to generate realistic stochastic scenarios 
based on historical load data.

Dependencies:
- numpy
- pandas
- matplotlib
- statsmodels
- tqdm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

# Load normalized time series data (values in [0, 1])
load_nl = np.load(Load_NL)

# Use the first 30 days (720 hours) for training
T = 24 * 30
data = load_nl[:T]

# Create a pandas Series with datetime index
index = pd.date_range(start='2023-01-01', periods=len(data), freq='H')
ts = pd.Series(data, index=index)

# Plot raw time series
plt.figure(figsize=(6, 3))
plt.plot(ts)
plt.title('Time Series Data')
plt.show()

# Perform Augmented Dickey-Fuller test for stationarity
#plot ADF test  https://www.projectpro.io/article/how-to-build-arima-model-in-python/544
results = adfuller(ts.values)
print('\n\n\t p-value is:', results[1])
print('\n\t data is stationary!!\n\n')

# Plot ACF and PACF to help identify ARIMA order
plot_acf(ts, lags=24)
plot_pacf(ts, lags=24)
plt.show()

# Define ARIMA order based on visual inspection (can be tuned)
p, d, q = 24, 0, 6
print(f'we choose the following attributes: \n\t p={p}, d={d}, q={q}\n')

# Fit the ARIMA model
model = ARIMA(ts, order=(p, d, q))
result = model.fit()

# Print model summary
print(result.summary())

# Residual diagnostics
residuals = pd.DataFrame(result.resid)

# Residual line plot
residuals.plot()
plt.title("Residuals")
plt.show()

# Residual distribution
residuals.plot(kind='kde')
plt.title("Residual Density")
plt.show()

# Residual statistics
print(residuals.describe())

# === Scenario Generation ===

anc = 24 * 27  # Start at hour 648 (day 27)
T = 24         # Simulate 24 hours ahead
NumSc = 100    # Number of valid scenarios to generate
LB = 0.7       # Lower bound for scenario filtering
UB = 1.3       # Upper bound for scenario filtering

# Generate forecast and extract smoothed state at anchor point
forecast = result.get_prediction(start=anc, end=anc+T, steps=T)
initial_state = result.smoothed_state[:, anc].T

# Simulate many scenarios to later filter out invalid ones
default_sim = result.simulate(
    nsimulations=T,
    repetitions=NumSc * 10,
    initial_state=initial_state,
    anchor=anc,
    random_state=0
).values

# Filter simulations to stay within [LB, UB] × true load
simulations = np.empty((T, 0))
for i in range(10 * NumSc):
    sim = default_sim[:, i].reshape(-1, 1)
    if all(LB * load_nl[anc + t] <= sim[t] <= UB * load_nl[anc + t] for t in range(T)):
        simulations = np.concatenate((simulations, sim), axis=1)
        if simulations.shape[1] >= NumSc:
            break

# Scale the simulated scenarios back to real-world values
sc_val = 17.246
simulations = simulations * sc_val

print(f"Generated {simulations.shape[1]} scenarios of length {T}")

# Calculate statistics
mean_values = simulations.mean(axis=1)
std_values = simulations.std(axis=1)

# Plot simulated scenarios with +-3sigma bounds
plt.figure(figsize=(5, 2.5))
plt.fill_between(
    range(anc, anc+T),
    mean_values - 3 * std_values,
    mean_values + 3 * std_values,
    color='grey', alpha=0.1, label='3$\sigma$'
)

# Plot all scenarios
for i in range(simulations.shape[1]):
    plt.plot(range(anc, anc+T), simulations[:, i], alpha=0.3)

# Plot true values and mean scenario
plt.plot(range(anc-12, anc+T+12), sc_val * load_nl[anc-12:anc+T+12], color='black', label='Truth')
plt.plot(range(anc, anc+T), mean_values, label='Avg-sim', linestyle='--', color='blue')

plt.xticks([anc, anc+T])
handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 2, 0]  # Reorder legend entries
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

plt.xlabel('Time (hour)')
plt.ylabel('Load Consumption (GW)')
plt.tight_layout()
plt.show()

# Plot standard deviation over time
plt.figure(figsize=(5, 3))
plt.plot(std_values, color='black', linestyle='--', label='std')
plt.legend()
plt.title("Scenario Standard Deviation")
plt.tight_layout()
plt.show()

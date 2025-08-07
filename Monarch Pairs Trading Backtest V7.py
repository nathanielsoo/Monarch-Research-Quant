import vectorbt as vbt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import adfuller
import ruptures as rpt


def test_cointegration(series_y, series_x, significance=0.05):
    pval1 = adfuller(series_x)[1]
    pval2 = adfuller(series_y)[1]
    if pval1 < significance or pval2 < significance:
        return False, None, None, None
    model = OLS(series_y, add_constant(series_x)).fit()
    resid = model.resid
    beta = model.params[1]
    adf_pval = adfuller(resid)[1]
    is_cointegrated = adf_pval < significance
    return is_cointegrated, beta, adf_pval, resid if is_cointegrated else None


def calculate_hurst_exponent(ts, start=2, end=20):
    ts = np.array(ts)
    if len(ts) < end:
        return np.nan
    lags = range(start, end)
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    log_lags = np.log(lags)
    log_tau = np.log(tau)
    slope, _ = np.polyfit(log_lags, log_tau, 1)
    return 2.0 * slope


def change_point(time_series):
    array = time_series.to_numpy()
    bkps = 7
    algo = rpt.Dynp(model="l2").fit(array)
    result = algo.predict(n_bkps=bkps)
    segments = [0] + result
    start_idx = segments[-2]
    end_idx = segments[-1]
    current_segment = array[start_idx:end_idx]
    segment_mean = np.mean(current_segment)
    segment_std = np.std(current_segment)
    current_value = array[-1]
    z_score = (current_value - segment_mean) / segment_std
    return abs(z_score)


def compute_favourable_regime_series(log_price, tickers, beta_window=60):
    regime_flags = pd.Series(False, index=log_price.index)
    for i in range(beta_window, len(log_price)):
        window = log_price.iloc[i - beta_window:i]
        is_cointegrated, beta, _, _ = test_cointegration(window[tickers[1]], window[tickers[0]])
        if not is_cointegrated or beta is None:
            continue
        spread_window = window[tickers[1]] - beta * window[tickers[0]]
        hurst = calculate_hurst_exponent(spread_window)
        try:
            cp = change_point(spread_window)
        except:
            cp = np.nan
        if is_cointegrated and hurst < 0.4 and cp < 2:
            regime_flags.iloc[i] = True
    return regime_flags.ffill().fillna(False)


asset_list_old = [["IVV", "VOO"], ["INTU", "MSFT"], ["ETN", "PH"], ["ABT", "ZTS"], ["DHR", "TMO"]]
tickers = asset_list_old[0]
start = datetime.today() - timedelta(days = 730)
end = datetime.today() - timedelta(days = 365)

price = vbt.YFData.download(tickers, start=start, end=end).get('Close').dropna()
log_price = np.log(price)

# Calculate regime series
favourable_regime_mask = compute_favourable_regime_series(log_price, tickers)

# Calculate beta once for spread calculation (use full period)
beta = test_cointegration(log_price[tickers[1]], log_price[tickers[0]])[1]
spread = log_price[tickers[1]] - beta * log_price[tickers[0]]

# Calculate z-score of spread
mean = spread.rolling(15).mean()
std = spread.rolling(15).std()
zscore = (spread - mean) / std

# Entry/exit signals
entries_short = (zscore >= 2.5) & favourable_regime_mask
exits_short = (zscore < 0.5) | (zscore >= 3)

entries_long = (zscore <= -2.5) & favourable_regime_mask
exits_long = (zscore > -0.5) | (zscore <= -3)

signal = pd.Series(0, index=price.index, dtype=int)
signal[entries_long] = 1
signal[entries_short] = -1
signal[exits_long & (signal.shift() == 1)] = 0
signal[exits_short & (signal.shift() == -1)] = 0
signal = signal.ffill().fillna(0)

# Construct portfolio weights
long_weights = np.array([-1 * beta, 1])
short_weights = np.array([1 * beta, -1])

weights = pd.DataFrame(index=price.index, columns=price.columns, dtype=float)
for i in range(len(price)):
    if signal.iloc[i] == 1:
        weights.iloc[i] = long_weights
    elif signal.iloc[i] == -1:
        weights.iloc[i] = short_weights
    else:
        weights.iloc[i] = [0, 0]

# Normalize weights
weights = weights.div(np.abs(weights).sum(axis=1), axis=0).fillna(0)

# Backtest
pf = vbt.Portfolio.from_orders(
    close=price,
    size=weights,
    size_type='targetpercent',
    direction='both',
    freq='1D',
    fees=0.005
)

print(pf.stats().to_frame())
(pf.value().sum(axis=1) / 2).vbt.plot(title="Total Portfolio Value").show()
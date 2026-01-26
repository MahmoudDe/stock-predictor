import numpy as np
import pandas as pd
import gc


def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band


def create_features(df, verbose=True, ticker_stats=None):
    """
    Create comprehensive features for stock prediction.
    
    Args:
        df: DataFrame with columns Date, Ticker, Open, High, Low, Close, Volume.
        verbose: If True, print progress.
        ticker_stats: Optional Series (Ticker -> value). For val/test, pass the
            result of train_df.groupby('Ticker')['ticker_mean_return_30d'].last()
            so Ticker-level features use train-only statistics (no leakage).
    
    OPTIMIZED VERSION with:
    - Reduced memory usage (float32)
    - Single calculation for MACD and Bollinger Bands (was 3x each)
    - Faster trend slope approximation
    - Garbage collection between stages
    
    Features are organized into three categories:
    1. Short-term patterns (local patterns)
    2. Seasonality
    3. Long-term behavior
    """
    def log(msg):
        if verbose:
            print(msg)
    
    df = df.copy()
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Convert to float32 early to reduce memory
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = df[col].astype('float32')
    
    grouped = df.groupby('Ticker', sort=False)
    feature_cols = []
    
    log("  [1/7] Basic price features...")
    # ============================================================
    # SHORT-TERM PATTERNS (Local Patterns)
    # ============================================================
    
    # 1. Basic Price Features
    df['price_change'] = grouped['Close'].diff()
    df['price_change_pct'] = grouped['Close'].pct_change()
    df['high_low_spread'] = df['High'] - df['Low']
    df['high_low_spread_pct'] = df['high_low_spread'] / (df['Close'] + 1e-8)
    df['open_close_diff'] = df['Close'] - df['Open']
    df['open_close_diff_pct'] = df['open_close_diff'] / (df['Open'] + 1e-8)
    feature_cols.extend(['price_change', 'price_change_pct', 'high_low_spread', 
                        'high_low_spread_pct', 'open_close_diff', 'open_close_diff_pct'])
    
    # 2. Price Momentum (1-day, 3-day, 5-day, 10-day returns)
    for days in [1, 3, 5, 10]:
        df[f'return_{days}d'] = grouped['Close'].pct_change(days)
        df[f'return_{days}d_abs'] = np.abs(df[f'return_{days}d'])
        feature_cols.extend([f'return_{days}d', f'return_{days}d_abs'])
    
    log("  [2/7] Technical indicators (RSI, MACD, Bollinger)...")
    # 3. Technical Indicators - OPTIMIZED: Calculate once per ticker
    # RSI
    df['rsi'] = grouped['Close'].transform(lambda x: calculate_rsi(x, window=14))
    feature_cols.append('rsi')
    
    # MACD - Calculate all components at once (FIXED: was calling 3x before)
    def compute_macd_all(prices):
        macd, signal, hist = calculate_macd(prices)
        return pd.DataFrame({'macd': macd, 'macd_signal': signal, 'macd_histogram': hist})
    
    macd_results = []
    for ticker, grp in grouped:
        result = compute_macd_all(grp['Close'])
        result.index = grp.index
        macd_results.append(result)
    macd_df = pd.concat(macd_results)
    df['macd'] = macd_df['macd']
    df['macd_signal'] = macd_df['macd_signal']
    df['macd_histogram'] = macd_df['macd_histogram']
    del macd_results, macd_df
    feature_cols.extend(['macd', 'macd_signal', 'macd_histogram'])
    
    # Bollinger Bands - Calculate all components at once (FIXED: was calling 3x before)
    def compute_bb_all(prices):
        upper, middle, lower = calculate_bollinger_bands(prices)
        return pd.DataFrame({'bb_upper': upper, 'bb_middle': middle, 'bb_lower': lower})
    
    bb_results = []
    for ticker, grp in grouped:
        result = compute_bb_all(grp['Close'])
        result.index = grp.index
        bb_results.append(result)
    bb_df = pd.concat(bb_results)
    df['bb_upper'] = bb_df['bb_upper']
    df['bb_middle'] = bb_df['bb_middle']
    df['bb_lower'] = bb_df['bb_lower']
    del bb_results, bb_df
    
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-8)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    feature_cols.extend(['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position'])
    
    gc.collect()
    
    log("  [3/7] Volume features...")
    # 4. Volume Features
    df['volume_change'] = grouped['Volume'].diff()
    df['volume_change_pct'] = grouped['Volume'].pct_change()
    df['volume_price_trend'] = df['volume_change_pct'] * df['price_change_pct']
    for window in [5, 7, 10, 30]:
        df[f'volume_ma_{window}'] = grouped['Volume'].transform(lambda x, w=window: x.rolling(w).mean())
        df[f'volume_to_ma_{window}'] = df['Volume'] / (df[f'volume_ma_{window}'] + 1e-8)
        feature_cols.extend([f'volume_ma_{window}', f'volume_to_ma_{window}'])
    feature_cols.extend(['volume_change', 'volume_change_pct', 'volume_price_trend'])
    
    # 5. Lagged Features (1, 2, 3, 5, 10 days)
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = grouped['Close'].shift(lag)
        df[f'close_lag_{lag}_pct'] = grouped['Close'].pct_change(lag)
        df[f'volume_lag_{lag}'] = grouped['Volume'].shift(lag)
        feature_cols.extend([f'close_lag_{lag}', f'close_lag_{lag}_pct', f'volume_lag_{lag}'])
    
    log("  [4/7] Rolling statistics...")
    # 6. Rolling Statistics (Short-term windows: 5, 10, 20 days)
    for window in [5, 10, 20]:
        df[f'close_std_{window}'] = grouped['Close'].transform(lambda x, w=window: x.rolling(w).std())
        df[f'close_mean_{window}'] = grouped['Close'].transform(lambda x, w=window: x.rolling(w).mean())
        df[f'close_min_{window}'] = grouped['Close'].transform(lambda x, w=window: x.rolling(w).min())
        df[f'close_max_{window}'] = grouped['Close'].transform(lambda x, w=window: x.rolling(w).max())
        df[f'close_range_{window}'] = df[f'close_max_{window}'] - df[f'close_min_{window}']
        df[f'close_range_pct_{window}'] = df[f'close_range_{window}'] / (df[f'close_mean_{window}'] + 1e-8)
        feature_cols.extend([f'close_std_{window}', f'close_mean_{window}', f'close_min_{window}', 
                            f'close_max_{window}', f'close_range_{window}', f'close_range_pct_{window}'])
    
    gc.collect()
    
    log("  [5/7] Seasonality features...")
    # ============================================================
    # SEASONALITY FEATURES
    # ============================================================
    
    # 7. Temporal Features
    df['day_of_week'] = df['Date'].dt.dayofweek.astype('int8')
    df['day_of_month'] = df['Date'].dt.day.astype('int8')
    df['month'] = df['Date'].dt.month.astype('int8')
    df['quarter'] = df['Date'].dt.quarter.astype('int8')
    df['year'] = df['Date'].dt.year.astype('int16')
    
    # Cyclical encoding (sin/cos) for day of week
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7).astype('float32')
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7).astype('float32')
    
    # Cyclical encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype('float32')
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype('float32')
    
    # Cyclical encoding for day of month
    df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31).astype('float32')
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31).astype('float32')
    
    # Trading day patterns
    df['days_since_month_start'] = df.groupby(['Ticker', df['Date'].dt.to_period('M')]).cumcount() + 1
    df['is_month_start'] = (df['day_of_month'] <= 3).astype('int8')
    df['is_month_end'] = (df['day_of_month'] >= 28).astype('int8')
    df['days_until_month_end'] = (31 - df['day_of_month']).astype('int8')
    
    feature_cols.extend(['day_of_week', 'day_of_month', 'month', 'quarter', 'year',
                        'day_of_week_sin', 'day_of_week_cos', 
                        'month_sin', 'month_cos',
                        'day_of_month_sin', 'day_of_month_cos',
                        'days_since_month_start', 'days_until_month_end', 
                        'is_month_start', 'is_month_end'])
    
    # Historical patterns (same day last year - approximate)
    df['year_ago_return'] = grouped['Close'].pct_change(252)
    feature_cols.append('year_ago_return')
    
    log("  [6/7] Long-term features (MA, volatility, trend)...")
    # ============================================================
    # LONG-TERM BEHAVIOR
    # ============================================================
    
    # 8. Long-term Moving Averages (50, 100, 200 days)
    for window in [50, 100, 200]:
        df[f'ma_{window}'] = grouped['Close'].transform(lambda x, w=window: x.rolling(w).mean())
        df[f'close_to_ma_{window}'] = df['Close'] / (df[f'ma_{window}'] + 1e-8) - 1
        df[f'ma_{window}_trend'] = grouped[f'ma_{window}'].diff() / (df[f'ma_{window}'] + 1e-8)
        feature_cols.extend([f'ma_{window}', f'close_to_ma_{window}', f'ma_{window}_trend'])
    
    # 9. Trend Indicators - OPTIMIZED: Use simple linear regression approximation
    # Instead of expensive rolling polyfit, use (last - first) / first as slope proxy
    for window in [30, 60, 90]:
        close_shifted = grouped['Close'].shift(window - 1)
        close_mean = grouped['Close'].transform(lambda x, w=window: x.rolling(w).mean())
        df[f'trend_slope_{window}'] = (df['Close'] - close_shifted) / (close_mean * window + 1e-8)
        feature_cols.append(f'trend_slope_{window}')
    
    # 10. Long-term Volatility Measures
    for window in [30, 60, 90]:
        df[f'volatility_{window}'] = grouped['Close'].transform(lambda x, w=window: x.rolling(w).std())
        df[f'volatility_pct_{window}'] = df[f'volatility_{window}'] / (df['Close'] + 1e-8)
        feature_cols.extend([f'volatility_{window}', f'volatility_pct_{window}'])
    
    # Volatility ratios
    df['volatility_ratio_30_60'] = df['volatility_30'] / (df['volatility_60'] + 1e-8)
    df['volatility_ratio_60_90'] = df['volatility_60'] / (df['volatility_90'] + 1e-8)
    feature_cols.extend(['volatility_ratio_30_60', 'volatility_ratio_60_90'])
    
    # 11. Price Position Features (relative to historical ranges)
    for window in [50, 100, 200]:
        df[f'price_min_{window}'] = grouped['Close'].transform(lambda x, w=window: x.rolling(w).min())
        df[f'price_max_{window}'] = grouped['Close'].transform(lambda x, w=window: x.rolling(w).max())
        df[f'price_position_{window}'] = (df['Close'] - df[f'price_min_{window}']) / (
            df[f'price_max_{window}'] - df[f'price_min_{window}'] + 1e-8
        )
        feature_cols.extend([f'price_min_{window}', f'price_max_{window}', f'price_position_{window}'])
    
    # 12. Long-term Momentum (30, 60, 90-day returns)
    for days in [30, 60, 90]:
        df[f'return_{days}d'] = grouped['Close'].pct_change(days)
        feature_cols.append(f'return_{days}d')
    
    # 12b. Ticker-level (cross-sectional): mean 30d return per Ticker
    # Train: expanding mean of return_30d per Ticker (no lookahead).
    # Val/Test: use last value from train per Ticker (passed via ticker_stats).
    if ticker_stats is None:
        df['ticker_mean_return_30d'] = grouped['return_30d'].transform(
            lambda x: x.expanding().mean().fillna(0)
        )
    else:
        df['ticker_mean_return_30d'] = df['Ticker'].map(ticker_stats).fillna(0)
    feature_cols.append('ticker_mean_return_30d')
    
    gc.collect()
    
    log("  [7/7] Market regime & interactions...")
    # ============================================================
    # MARKET REGIME FEATURES
    # ============================================================
    
    # 13. Market Regime Indicators
    df['is_bull_market'] = (df['Close'] > df['ma_200']).astype('int8')
    df['is_bear_market'] = (df['Close'] < df['ma_200']).astype('int8')
    df['market_regime_strength'] = np.abs(df['close_to_ma_200'])
    
    # Volatility regime
    df['is_high_volatility'] = (df['volatility_30'] > df['volatility_60']).astype('int8')
    df['volatility_regime'] = df['volatility_ratio_30_60']
    
    # Trend strength
    df['trend_strength_30'] = np.abs(df['trend_slope_30'])
    df['trend_strength_60'] = np.abs(df['trend_slope_60'])
    df['trend_direction'] = np.sign(df['trend_slope_30'].fillna(0)).astype('int8')
    
    feature_cols.extend(['is_bull_market', 'is_bear_market', 'market_regime_strength',
                        'is_high_volatility', 'volatility_regime',
                        'trend_strength_30', 'trend_strength_60', 'trend_direction'])
    
    # ============================================================
    # FEATURE INTERACTIONS
    # ============================================================
    
    # 14. Key Feature Interactions
    df['rsi_volume_interaction'] = df['rsi'] * df['volume_to_ma_10']
    df['macd_volatility_interaction'] = df['macd'] * df['volatility_30']
    df['momentum_volume_interaction'] = df['return_10d'] * df['volume_change_pct']
    df['rsi_trend_interaction'] = df['rsi'] * df['trend_slope_30']
    df['macd_price_position_interaction'] = df['macd'] * df['price_position_100']
    df['volatility_volume_interaction'] = df['volatility_30'] * df['volume_to_ma_10']
    df['bb_position_volume_interaction'] = df['bb_position'] * df['volume_to_ma_10']
    df['trend_volatility_interaction'] = df['trend_slope_30'] * df['volatility_30']
    
    feature_cols.extend(['rsi_volume_interaction', 'macd_volatility_interaction',
                        'momentum_volume_interaction', 'rsi_trend_interaction',
                        'macd_price_position_interaction', 'volatility_volume_interaction',
                        'bb_position_volume_interaction', 'trend_volatility_interaction'])
    
    # ============================================================
    # CLEANUP - More efficient than looping twice
    # ============================================================
    log("  Cleaning up NaN/Inf values...")
    
    # Fill NaN and replace inf in one pass per column
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = grouped[col].transform(lambda x: x.ffill().bfill().fillna(0))
            # Convert to float32 to save memory
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
    
    gc.collect()
    log("  Done!")
    
    return df, feature_cols

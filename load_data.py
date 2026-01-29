import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class StockDataLoader:
  
    def __init__(self, data_dir: str = "data", train_file: str = "train.csv"):
        self.data_dir = Path(data_dir)
        self.train_file = self.data_dir / train_file
        self.prediction_horizon = 30
        
    def load_data(self) -> pd.DataFrame:
        print("Loading training data...")
        print(f"Reading from: {self.train_file}")
        
        # Load data in chunks if it's very large
        try:
            # Try to load all at once first
            df = pd.read_csv(self.train_file)
            print(f"✓ Loaded {len(df):,} rows")
        except MemoryError:
            print("File too large, loading in chunks...")
            chunk_list = []
            chunk_size = 1000000
            for chunk in pd.read_csv(self.train_file, chunksize=chunk_size):
                chunk_list.append(chunk)
            df = pd.concat(chunk_list, ignore_index=True)
            print(f"✓ Loaded {len(df):,} rows in chunks")
        
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by Ticker and Date to ensure proper time series order
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        print(f"✓ Data loaded: {len(df):,} rows, {df['Ticker'].nunique()} unique tickers")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\nHandling missing values...")
        
        initial_rows = len(df)
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print("Missing values found:")
            print(missing_counts[missing_counts > 0])
        else:
            print("✓ No missing values found")
        
        # Handle missing values in price columns (Open, High, Low, Close)
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if df[col].isnull().sum() > 0:
                # Forward fill within each ticker (use previous day's value)
                df[col] = df.groupby('Ticker')[col].ffill()
                # If still missing (e.g., first day), use backward fill
                df[col] = df.groupby('Ticker')[col].bfill()
                # If still missing, fill with 0 (shouldn't happen for valid data)
                df[col] = df[col].fillna(0)
        
        # Handle missing values in Volume
        if df['Volume'].isnull().sum() > 0:
            df['Volume'] = df.groupby('Ticker')['Volume'].ffill()
            df['Volume'] = df['Volume'].bfill()
            df['Volume'] = df['Volume'].fillna(0)
        
        # Handle missing values in Dividends and Stock Splits
        df['Dividends'] = df['Dividends'].fillna(0)
        df['Stock Splits'] = df['Stock Splits'].fillna(0)
        
        # Remove rows where Close price is 0 or negative (invalid data)
        invalid_price_mask = (df['Close'] <= 0) | (df['Open'] <= 0) | (df['High'] <= 0) | (df['Low'] <= 0)
        if invalid_price_mask.sum() > 0:
            print(f"  Removing {invalid_price_mask.sum():,} rows with invalid prices")
            df = df[~invalid_price_mask].reset_index(drop=True)
        
        # Remove tickers with insufficient data (need at least prediction_horizon + 1 days)
        ticker_counts = df.groupby('Ticker').size()
        valid_tickers = ticker_counts[ticker_counts >= self.prediction_horizon + 1].index
        df = df[df['Ticker'].isin(valid_tickers)].reset_index(drop=True)
        
        final_rows = len(df)
        removed = initial_rows - final_rows
        if removed > 0:
            print(f"✓ Removed {removed:,} rows with invalid data")
        print(f"✓ Final dataset: {final_rows:,} rows, {df['Ticker'].nunique()} unique tickers")
        
        return df
    
    def create_target_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target labels: predict if Close price after 30 trading days will be ↑ (higher) or ↓ (lower).
        
        Args:
            df: DataFrame with stock data sorted by Ticker and Date
            
        Returns:
            DataFrame with added 'target' column (1 for ↑, 0 for ↓)
        """
        print(f"\nCreating target labels (prediction horizon: {self.prediction_horizon} trading days)...")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # For each row, look ahead prediction_horizon days to get the future Close price
        # Group by Ticker to ensure we only look ahead within the same stock
        df['future_close'] = df.groupby('Ticker')['Close'].shift(-self.prediction_horizon)
        
        # Create target: 1 if future_close > current_close (↑), 0 if future_close <= current_close (↓)
        df['target'] = (df['future_close'] > df['Close']).astype(int)
        
        # Remove rows where we don't have future_close (last prediction_horizon days for each ticker)
        initial_rows = len(df)
        df = df.dropna(subset=['future_close']).reset_index(drop=True)
        removed = initial_rows - len(df)
        
        print(f"✓ Created target labels")
        print(f"  Removed {removed:,} rows without future prices (last {self.prediction_horizon} days per ticker)")
        print(f"  Final dataset: {len(df):,} rows")
        
        # Show target distribution
        target_dist = df['target'].value_counts()
        print(f"  Target distribution:")
        print(f"    ↑ (Higher): {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df)*100:.2f}%)")
        print(f"    ↓ (Lower): {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df)*100:.2f}%)")
        
        # Drop the temporary future_close column
        df = df.drop(columns=['future_close'])
        
        return df
    
    def split_time_series(self, df: pd.DataFrame, 
                         train_ratio: float = 0.7, 
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print(f"\nSplitting data into train/validation/test sets...")
        print(f"  Ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
        
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        unique_dates = df['Date'].unique()
        unique_dates = np.sort(unique_dates)
        
        n_dates = len(unique_dates)
        train_end_idx = int(n_dates * train_ratio)
        val_end_idx = int(n_dates * (train_ratio + val_ratio))
        
        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[train_end_idx:val_end_idx]
        test_dates = unique_dates[val_end_idx:]
        
        # Split data based on dates
        train_df = df[df['Date'].isin(train_dates)].copy().reset_index(drop=True)
        val_df = df[df['Date'].isin(val_dates)].copy().reset_index(drop=True)
        test_df = df[df['Date'].isin(test_dates)].copy().reset_index(drop=True)
        
        print(f"✓ Data split completed:")
        print(f"  Train: {len(train_df):,} rows ({len(train_df)/len(df)*100:.2f}%)")
        print(f"    Date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
        print(f"  Validation: {len(val_df):,} rows ({len(val_df)/len(df)*100:.2f}%)")
        print(f"    Date range: {val_df['Date'].min()} to {val_df['Date'].max()}")
        print(f"  Test: {len(test_df):,} rows ({len(test_df)/len(df)*100:.2f}%)")
        print(f"    Date range: {test_df['Date'].min()} to {test_df['Date'].max()}")
        
        # Show target distribution in each split
        print(f"\n  Target distribution by split:")
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            if len(split_df) > 0:
                target_dist = split_df['target'].value_counts()
                up_pct = target_dist.get(1, 0) / len(split_df) * 100
                down_pct = target_dist.get(0, 0) / len(split_df) * 100
                print(f"    {split_name}: ↑ {target_dist.get(1, 0):,} ({up_pct:.2f}%), "
                      f"↓ {target_dist.get(0, 0):,} ({down_pct:.2f}%)")
        
        return train_df, val_df, test_df
    
    def get_feature_columns(self) -> list:
        return ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    
    def process(self, train_ratio: float = 0.7, 
                val_ratio: float = 0.15,
                test_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        print("=" * 60)
        print("STOCK DATA LOADING AND PREPROCESSING")
        print("=" * 60)
        

        df = self.load_data()
        
        df = self.handle_missing_values(df)
        
        df = self.create_target_labels(df)
        
        train_df, val_df, test_df = self.split_time_series(df, train_ratio, val_ratio, test_ratio)
        
        print("\n" + "=" * 60)
        print("✓ DATA PROCESSING COMPLETE")
        print("=" * 60)
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'full': df
        }


def main():
    loader = StockDataLoader()
    # Process data
    datasets = loader.process(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Optionally save processed data
    save_processed = input("\nSave processed datasets? (y/n): ").strip().lower()
    if save_processed == 'y':
        output_dir = Path("data/processed")
        output_dir.mkdir(exist_ok=True)
        
        for split_name, df in datasets.items():
            if split_name != 'full':
                output_file = output_dir / f"{split_name}.csv"
                df.to_csv(output_file, index=False)
                print(f"✓ Saved {split_name} dataset to {output_file}")
    
    return datasets


if __name__ == "__main__":
    datasets = main()


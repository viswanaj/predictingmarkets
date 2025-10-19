#!/usr/bin/env python3
"""
High-Frequency Stock Data Downloader
Downloads stock data at various intraday intervals for detailed analysis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import List, Dict, Optional, Tuple
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('high_freq_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HighFrequencyDownloader:
    """
    A class to download high-frequency stock market data.
    """
    
    def __init__(self, data_dir: str = "high_freq_data"):
        """
        Initialize the high-frequency downloader.
        
        Args:
            data_dir: Directory to save downloaded data
        """
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # Stock tickers mapping
        self.stock_tickers = {
            "Novo Nordisk": "NVO",
            "Eli Lilly": "LLY", 
            "Apple": "AAPL",
            "NVIDIA": "NVDA",
            "Genentech": "DNA"
        }
        
        # Available intervals and their characteristics
        self.intervals = {
            '1m': {'name': '1 Minute', 'max_period': '7d', 'records_per_day': 390},
            '2m': {'name': '2 Minutes', 'max_period': '60d', 'records_per_day': 195},
            '5m': {'name': '5 Minutes', 'max_period': '60d', 'records_per_day': 78},
            '15m': {'name': '15 Minutes', 'max_period': '60d', 'records_per_day': 26},
            '30m': {'name': '30 Minutes', 'max_period': '60d', 'records_per_day': 13},
            '60m': {'name': '1 Hour', 'max_period': '730d', 'records_per_day': 7},
            '90m': {'name': '90 Minutes', 'max_period': '60d', 'records_per_day': 5},
            '1h': {'name': '1 Hour', 'max_period': '730d', 'records_per_day': 7}
        }
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created high-frequency data directory: {self.data_dir}")
    
    def download_high_frequency_data(self, ticker: str, interval: str, 
                                   period: str = None) -> Optional[pd.DataFrame]:
        """
        Download high-frequency data for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h)
            period: Time period (if None, uses max available for interval)
            
        Returns:
            DataFrame with stock data or None if failed
        """
        try:
            # Use max period if not specified
            if period is None:
                period = self.intervals[interval]['max_period']
            
            logger.info(f"Downloading {interval} data for {ticker} (period: {period})...")
            
            # Create yfinance ticker object
            stock = yf.Ticker(ticker)
            
            # Download historical data
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {ticker} at {interval} interval")
                return None
            
            # Add ticker column
            data['Ticker'] = ticker
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Rename columns for consistency
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Ticker']
            
            # Add additional calculated columns
            data['Price_Change'] = data['Close'] - data['Open']
            data['Price_Change_Pct'] = (data['Price_Change'] / data['Open']) * 100
            
            # Calculate returns based on interval
            data['Return'] = data['Close'].pct_change()
            
            # Add interval-specific calculations
            if interval in ['1m', '2m', '5m']:
                # For very short intervals, calculate minute-based metrics
                data['Minute_Return'] = data['Return'] * (60 / self._get_minutes_per_interval(interval))
            else:
                data['Minute_Return'] = data['Return']
            
            # Volume metrics
            data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']
            
            # Price volatility (rolling standard deviation)
            data['Price_Volatility'] = data['Return'].rolling(window=20).std()
            
            logger.info(f"Successfully downloaded {len(data)} records for {ticker} at {interval} interval")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading {interval} data for {ticker}: {str(e)}")
            return None
    
    def _get_minutes_per_interval(self, interval: str) -> int:
        """Get number of minutes per interval."""
        interval_map = {
            '1m': 1, '2m': 2, '5m': 5, '15m': 15, 
            '30m': 30, '60m': 60, '90m': 90, '1h': 60
        }
        return interval_map.get(interval, 1)
    
    def download_all_intervals(self, ticker: str, intervals: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Download data for a single stock at multiple intervals.
        
        Args:
            ticker: Stock ticker symbol
            intervals: List of intervals to download (if None, downloads all)
            
        Returns:
            Dictionary mapping intervals to DataFrames
        """
        if intervals is None:
            intervals = list(self.intervals.keys())
        
        results = {}
        
        for interval in intervals:
            if interval not in self.intervals:
                logger.warning(f"Unknown interval: {interval}")
                continue
                
            data = self.download_high_frequency_data(ticker, interval)
            if data is not None:
                results[interval] = data
                # Add delay to respect rate limits
                time.sleep(1)
            else:
                logger.warning(f"Skipping {interval} for {ticker} due to download failure")
        
        return results
    
    def download_all_stocks_high_freq(self, intervals: List[str] = None, 
                                   period: str = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Download high-frequency data for all configured stocks.
        
        Args:
            intervals: List of intervals to download
            period: Time period for data
            
        Returns:
            Nested dictionary: {ticker: {interval: DataFrame}}
        """
        if intervals is None:
            intervals = ['5m', '15m', '30m', '1h']  # Default to reasonable intervals
        
        all_results = {}
        tickers = list(self.stock_tickers.values())
        
        logger.info(f"Starting high-frequency download for {len(tickers)} stocks at {len(intervals)} intervals")
        
        for ticker in tickers:
            logger.info(f"Processing {ticker}...")
            ticker_results = {}
            
            for interval in intervals:
                data = self.download_high_frequency_data(ticker, interval, period)
                if data is not None:
                    ticker_results[interval] = data
                    # Save individual files
                    self.save_interval_data(data, ticker, interval)
                
                # Add delay between requests
                time.sleep(2)
            
            all_results[ticker] = ticker_results
            
            # Longer delay between stocks
            time.sleep(5)
        
        # Save combined data
        self.save_combined_high_freq_data(all_results)
        
        return all_results
    
    def save_interval_data(self, data: pd.DataFrame, ticker: str, interval: str):
        """Save data for a specific ticker and interval."""
        # Create interval-specific directory
        interval_dir = os.path.join(self.data_dir, interval)
        if not os.path.exists(interval_dir):
            os.makedirs(interval_dir)
        
        # Save in multiple formats
        filename_base = f"{ticker}_{interval}"
        
        # CSV
        csv_path = os.path.join(interval_dir, f"{filename_base}.csv")
        data.to_csv(csv_path, index=False)
        
        # Parquet (more efficient for large datasets)
        parquet_path = os.path.join(interval_dir, f"{filename_base}.parquet")
        data.to_parquet(parquet_path, index=False)
        
        logger.info(f"Saved {interval} data for {ticker}: {len(data)} records")
    
    def save_combined_high_freq_data(self, all_results: Dict[str, Dict[str, pd.DataFrame]]):
        """Save combined high-frequency data."""
        for interval in self.intervals.keys():
            interval_data = []
            
            for ticker, ticker_results in all_results.items():
                if interval in ticker_results:
                    interval_data.append(ticker_results[interval])
            
            if interval_data:
                # Combine all stocks for this interval
                combined_data = pd.concat(interval_data, ignore_index=True)
                combined_data = combined_data.sort_values(['Date', 'Ticker'])
                
                # Save combined data
                interval_dir = os.path.join(self.data_dir, interval)
                combined_path = os.path.join(interval_dir, f"all_stocks_{interval}.csv")
                combined_data.to_csv(combined_path, index=False)
                
                parquet_path = os.path.join(interval_dir, f"all_stocks_{interval}.parquet")
                combined_data.to_parquet(parquet_path, index=False)
                
                logger.info(f"Saved combined {interval} data: {len(combined_data)} records")
    
    def create_summary_report(self, all_results: Dict[str, Dict[str, pd.DataFrame]]):
        """Create a summary report of downloaded data."""
        summary_data = []
        
        for ticker, ticker_results in all_results.items():
            for interval, data in ticker_results.items():
                if not data.empty:
                    summary_data.append({
                        'Ticker': ticker,
                        'Interval': interval,
                        'Interval_Name': self.intervals[interval]['name'],
                        'Records': len(data),
                        'Date_Range_Start': data['Date'].min(),
                        'Date_Range_End': data['Date'].max(),
                        'Avg_Volume': data['Volume'].mean(),
                        'Price_Range_Min': data['Close'].min(),
                        'Price_Range_Max': data['Close'].max(),
                        'Volatility': data['Return'].std() * 100
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_path = os.path.join(self.data_dir, "high_freq_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Save metadata
        metadata = {
            "download_timestamp": datetime.now().isoformat(),
            "total_intervals": len(self.intervals),
            "total_stocks": len(self.stock_tickers),
            "intervals_downloaded": list(set([item['Interval'] for item in summary_data])),
            "total_records": summary_df['Records'].sum()
        }
        
        metadata_path = os.path.join(self.data_dir, "high_freq_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Summary report saved: {len(summary_data)} datasets")
        return summary_df

def main():
    """Main function to run the high-frequency downloader."""
    try:
        # Initialize downloader
        downloader = HighFrequencyDownloader()
        
        print("🚀 High-Frequency Stock Data Downloader")
        print("="*60)
        
        # Choose intervals to download
        print("\nAvailable intervals:")
        for interval, info in downloader.intervals.items():
            print(f"  {interval}: {info['name']} (max period: {info['max_period']}, ~{info['records_per_day']} records/day)")
        
        # Download data for multiple intervals
        intervals_to_download = ['5m', '15m', '30m', '1h']  # Reasonable balance of detail vs. data size
        
        print(f"\n📥 Downloading data for intervals: {', '.join(intervals_to_download)}")
        print("This may take several minutes due to rate limiting...")
        
        # Download all data
        all_results = downloader.download_all_stocks_high_freq(intervals=intervals_to_download)
        
        # Create summary report
        summary = downloader.create_summary_report(all_results)
        
        print("\n" + "="*60)
        print("🎉 HIGH-FREQUENCY DATA DOWNLOAD COMPLETED!")
        print("="*60)
        
        print(f"\n📊 Summary:")
        print(f"  Stocks processed: {len(downloader.stock_tickers)}")
        print(f"  Intervals downloaded: {len(intervals_to_download)}")
        print(f"  Total datasets: {len(summary)}")
        print(f"  Total records: {summary['Records'].sum():,}")
        
        print(f"\n📁 Data saved in: {downloader.data_dir}/")
        print("\nDirectory structure:")
        print("  high_freq_data/")
        for interval in intervals_to_download:
            print(f"    ├── {interval}/")
            print(f"    │   ├── AAPL_{interval}.csv")
            print(f"    │   ├── AAPL_{interval}.parquet")
            print(f"    │   ├── ... (other stocks)")
            print(f"    │   └── all_stocks_{interval}.csv")
        
        print(f"\n📈 Use cases for high-frequency data:")
        print(f"  - Intraday trading strategies")
        print(f"  - Market microstructure analysis")
        print(f"  - Volatility modeling")
        print(f"  - High-frequency trading algorithms")
        print(f"  - Real-time market prediction")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()

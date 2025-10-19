#!/usr/bin/env python3
"""
Stock Market Data Downloader
Downloads historical stock data for specified tickers and saves in indexable formats.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import List, Dict, Optional, Tuple
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockDataDownloader:
    """
    A class to download and process stock market data from Yahoo Finance.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the downloader.
        
        Args:
            data_dir: Directory to save downloaded data
        """
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # Stock tickers mapping (some may need adjustment for Yahoo Finance)
        self.stock_tickers = {
            "Novo Nordisk": "NVO",      # Danish company, trades on NYSE
            "Eli Lilly": "LLY",         # NYSE
            "Apple": "AAPL",           # NASDAQ
            "NVIDIA": "NVDA",          # NASDAQ
            "Genentech": "DNA"         # Roche subsidiary, trades as DNA
        }
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def download_single_stock(self, ticker: str, period: str = "2y", 
                            interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Download data for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with stock data or None if failed
        """
        try:
            logger.info(f"Downloading data for {ticker}...")
            
            # Create yfinance ticker object
            stock = yf.Ticker(ticker)
            
            # Download historical data
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return None
            
            # Add ticker column
            data['Ticker'] = ticker
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Rename columns for consistency
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Ticker']
            
            # Add additional calculated columns
            data['Daily_Return'] = data['Close'].pct_change()
            data['Price_Change'] = data['Close'] - data['Open']
            data['Price_Change_Pct'] = (data['Price_Change'] / data['Open']) * 100
            
            logger.info(f"Successfully downloaded {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {str(e)}")
            return None
    
    def download_multiple_stocks(self, tickers: List[str], period: str = "2y", 
                               interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple stocks.
        
        Args:
            tickers: List of stock ticker symbols
            period: Time period for data
            interval: Data interval
            
        Returns:
            Dictionary mapping tickers to DataFrames
        """
        results = {}
        
        for ticker in tickers:
            data = self.download_single_stock(ticker, period, interval)
            if data is not None:
                results[ticker] = data
            else:
                logger.warning(f"Skipping {ticker} due to download failure")
        
        return results
    
    def save_data_csv(self, data: pd.DataFrame, filename: str):
        """
        Save DataFrame to CSV file.
        
        Args:
            data: DataFrame to save
            filename: Output filename
        """
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    def save_data_json(self, data: pd.DataFrame, filename: str):
        """
        Save DataFrame to JSON file.
        
        Args:
            data: DataFrame to save
            filename: Output filename
        """
        filepath = os.path.join(self.data_dir, filename)
        data.to_json(filepath, orient='records', date_format='iso')
        logger.info(f"Data saved to {filepath}")
    
    def save_data_parquet(self, data: pd.DataFrame, filename: str):
        """
        Save DataFrame to Parquet file (efficient for large datasets).
        
        Args:
            data: DataFrame to save
            filename: Output filename
        """
        filepath = os.path.join(self.data_dir, filename)
        data.to_parquet(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    def combine_all_data(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine data from all stocks into a single DataFrame.
        
        Args:
            stock_data: Dictionary of stock data
            
        Returns:
            Combined DataFrame
        """
        if not stock_data:
            logger.warning("No stock data to combine")
            return pd.DataFrame()
        
        # Combine all DataFrames
        combined_data = pd.concat(stock_data.values(), ignore_index=True)
        
        # Sort by Date and Ticker
        combined_data = combined_data.sort_values(['Date', 'Ticker'])
        
        logger.info(f"Combined data from {len(stock_data)} stocks into {len(combined_data)} total records")
        return combined_data
    
    def create_summary_stats(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create summary statistics for all stocks.
        
        Args:
            stock_data: Dictionary of stock data
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for ticker, data in stock_data.items():
            if data.empty:
                continue
                
            latest_price = data['Close'].iloc[-1]
            price_change_1d = data['Daily_Return'].iloc[-1] * 100 if len(data) > 1 else 0
            price_change_1w = ((data['Close'].iloc[-1] / data['Close'].iloc[-6]) - 1) * 100 if len(data) >= 6 else 0
            avg_volume = data['Volume'].mean()
            volatility = data['Daily_Return'].std() * 100
            
            summary_data.append({
                'Ticker': ticker,
                'Latest_Price': latest_price,
                'Price_Change_1D_Pct': price_change_1d,
                'Price_Change_1W_Pct': price_change_1w,
                'Avg_Volume': avg_volume,
                'Volatility_Pct': volatility,
                'Data_Points': len(data),
                'Date_Range_Start': data['Date'].min(),
                'Date_Range_End': data['Date'].max()
            })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def download_and_save_all(self, period: str = "2y", interval: str = "1d"):
        """
        Download data for all configured stocks and save in multiple formats.
        
        Args:
            period: Time period for data
            interval: Data interval
        """
        logger.info("Starting comprehensive stock data download...")
        
        # Get ticker symbols
        tickers = list(self.stock_tickers.values())
        ticker_names = list(self.stock_tickers.keys())
        
        logger.info(f"Downloading data for: {', '.join([f'{name} ({ticker})' for name, ticker in self.stock_tickers.items()])}")
        
        # Download data for all stocks
        stock_data = self.download_multiple_stocks(tickers, period, interval)
        
        if not stock_data:
            logger.error("No data was successfully downloaded")
            return
        
        # Save individual stock files
        for ticker, data in stock_data.items():
            # Find the company name for this ticker
            company_name = next(name for name, t in self.stock_tickers.items() if t == ticker)
            
            # Save in multiple formats
            self.save_data_csv(data, f"{ticker}_{company_name.replace(' ', '_')}_data.csv")
            self.save_data_json(data, f"{ticker}_{company_name.replace(' ', '_')}_data.json")
        
        # Combine all data
        combined_data = self.combine_all_data(stock_data)
        if not combined_data.empty:
            self.save_data_csv(combined_data, "all_stocks_combined.csv")
            self.save_data_json(combined_data, "all_stocks_combined.json")
            self.save_data_parquet(combined_data, "all_stocks_combined.parquet")
        
        # Create and save summary statistics
        summary_stats = self.create_summary_stats(stock_data)
        if not summary_stats.empty:
            self.save_data_csv(summary_stats, "stock_summary_stats.csv")
            self.save_data_json(summary_stats, "stock_summary_stats.json")
        
        # Save metadata
        metadata = {
            "download_timestamp": datetime.now().isoformat(),
            "period": period,
            "interval": interval,
            "stocks": self.stock_tickers,
            "successful_downloads": list(stock_data.keys()),
            "total_records": len(combined_data) if not combined_data.empty else 0
        }
        
        metadata_file = os.path.join(self.data_dir, "download_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Stock data download completed successfully!")
        logger.info(f"Data saved in directory: {self.data_dir}")
        logger.info(f"Successfully downloaded data for {len(stock_data)} stocks")

def main():
    """Main function to run the stock data downloader."""
    try:
        # Initialize downloader
        downloader = StockDataDownloader()
        
        # Download data for all configured stocks
        downloader.download_and_save_all(period="2y", interval="1d")
        
        print("\n" + "="*60)
        print("STOCK DATA DOWNLOAD COMPLETED!")
        print("="*60)
        print(f"Data saved in: {downloader.data_dir}/")
        print("\nFiles created:")
        
        # List created files
        if os.path.exists(downloader.data_dir):
            files = os.listdir(downloader.data_dir)
            for file in sorted(files):
                print(f"  - {file}")
        
        print("\nYou can now use these files for:")
        print("  - CSV files: Excel, pandas, R, etc.")
        print("  - JSON files: Web applications, APIs")
        print("  - Parquet files: Big data processing (pandas, Spark)")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()

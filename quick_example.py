#!/usr/bin/env python3
"""
Quick Stock Data Example
Simple example showing how to use the stock data downloader and processor.
"""

from stock_data_downloader import StockDataDownloader
from stock_data_processor import StockDataProcessor
import pandas as pd

def quick_example():
    """Run a quick example of downloading and processing stock data."""
    
    print("🚀 Starting Quick Stock Data Example")
    print("="*50)
    
    # Step 1: Download stock data
    print("\n📥 Step 1: Downloading stock data...")
    downloader = StockDataDownloader()
    downloader.download_and_save_all(period="1y", interval="1d")
    
    # Step 2: Process the data
    print("\n📊 Step 2: Processing stock data...")
    processor = StockDataProcessor()
    data = processor.load_combined_data()
    
    # Step 3: Show basic statistics
    print("\n📈 Step 3: Basic Statistics")
    print("-" * 30)
    
    for ticker in data['Ticker'].unique():
        ticker_data = data[data['Ticker'] == ticker]
        latest_price = ticker_data['Close'].iloc[-1]
        total_return = ((ticker_data['Close'].iloc[-1] / ticker_data['Close'].iloc[0]) - 1) * 100
        volatility = ticker_data['Daily_Return'].std() * 100
        
        print(f"{ticker}:")
        print(f"  Latest Price: ${latest_price:.2f}")
        print(f"  Total Return (1Y): {total_return:.2f}%")
        print(f"  Volatility: {volatility:.2f}%")
        print()
    
    # Step 4: Show correlation
    print("🔗 Step 4: Stock Correlations")
    print("-" * 30)
    correlation_matrix = processor.calculate_correlation_matrix(data)
    print(correlation_matrix.round(3))
    
    print("\n✅ Example completed successfully!")
    print("Check the 'data/' and 'analysis/' directories for detailed results.")

if __name__ == "__main__":
    quick_example()

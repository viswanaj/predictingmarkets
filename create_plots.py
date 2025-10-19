#!/usr/bin/env python3
"""
Stock Price Visualization
Creates subplots showing closing prices for each stock over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import seaborn as sns
from stock_data_processor import StockDataProcessor

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_stock_price_plots():
    """Create subplots showing closing prices for each stock."""
    
    # Load the combined data
    processor = StockDataProcessor()
    data = processor.load_combined_data()
    
    # Get unique tickers and their company names
    tickers = data['Ticker'].unique()
    company_names = {
        'AAPL': 'Apple Inc.',
        'NVDA': 'NVIDIA Corporation', 
        'LLY': 'Eli Lilly and Company',
        'NVO': 'Novo Nordisk A/S',
        'DNA': 'Genentech (Roche)'
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(tickers), 1, figsize=(15, 20))
    fig.suptitle('Stock Closing Prices Over Time', fontsize=20, fontweight='bold', y=0.98)
    
    # If only one ticker, make axes a list
    if len(tickers) == 1:
        axes = [axes]
    
    # Plot each stock
    for i, ticker in enumerate(tickers):
        # Filter data for this ticker
        ticker_data = data[data['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date')
        
        # Get company name
        company_name = company_names.get(ticker, ticker)
        
        # Plot closing price
        axes[i].plot(ticker_data['Date'], ticker_data['Close'], 
                    linewidth=2, color=f'C{i}', alpha=0.8)
        
        # Add moving averages
        ticker_data['MA_20'] = ticker_data['Close'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['Close'].rolling(window=50).mean()
        
        axes[i].plot(ticker_data['Date'], ticker_data['MA_20'], 
                    '--', alpha=0.7, linewidth=1, color='orange', label='20-day MA')
        axes[i].plot(ticker_data['Date'], ticker_data['MA_50'], 
                    '--', alpha=0.7, linewidth=1, color='red', label='50-day MA')
        
        # Formatting
        axes[i].set_title(f'{company_name} ({ticker})', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Price ($)', fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper left', fontsize=10)
        
        # Format x-axis
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[i].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
        
        # Add price range annotation
        min_price = ticker_data['Close'].min()
        max_price = ticker_data['Close'].max()
        price_range = max_price - min_price
        
        axes[i].annotate(f'Range: ${min_price:.2f} - ${max_price:.2f}\n'
                        f'Latest: ${ticker_data["Close"].iloc[-1]:.2f}',
                        xy=(0.02, 0.98), xycoords='axes fraction',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                        fontsize=10, verticalalignment='top')
    
    # Format the last subplot's x-axis label
    axes[-1].set_xlabel('Date', fontsize=12)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the plot
    plt.savefig('stock_prices_subplots.png', dpi=300, bbox_inches='tight')
    plt.savefig('stock_prices_subplots.pdf', bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    print("✅ Stock price plots created successfully!")
    print("📁 Files saved:")
    print("   - stock_prices_subplots.png (high resolution)")
    print("   - stock_prices_subplots.pdf (vector format)")

def create_combined_price_plot():
    """Create a single plot with all stocks normalized to starting price."""
    
    # Load the combined data
    processor = StockDataProcessor()
    data = processor.load_combined_data()
    
    # Company names for legend
    company_names = {
        'AAPL': 'Apple Inc.',
        'NVDA': 'NVIDIA Corporation', 
        'LLY': 'Eli Lilly and Company',
        'NVO': 'Novo Nordisk A/S',
        'DNA': 'Genentech (Roche)'
    }
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot each stock normalized to starting price
    for ticker in data['Ticker'].unique():
        ticker_data = data[data['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date')
        
        # Normalize to starting price (first price = 100)
        normalized_price = (ticker_data['Close'] / ticker_data['Close'].iloc[0]) * 100
        
        company_name = company_names.get(ticker, ticker)
        plt.plot(ticker_data['Date'], normalized_price, 
                linewidth=2, label=f'{company_name} ({ticker})', alpha=0.8)
    
    # Formatting
    plt.title('Stock Performance Comparison (Normalized to Starting Price = 100)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Normalized Price (Starting Price = 100)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Add horizontal line at 100
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('stock_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('stock_performance_comparison.pdf', bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    print("✅ Combined performance plot created successfully!")
    print("📁 Files saved:")
    print("   - stock_performance_comparison.png (high resolution)")
    print("   - stock_performance_comparison.pdf (vector format)")

def main():
    """Main function to create all plots."""
    try:
        print("🎨 Creating stock price visualizations...")
        print("="*50)
        
        # Create individual subplots
        print("\n📊 Creating individual stock subplots...")
        create_stock_price_plots()
        
        # Create combined comparison plot
        print("\n📈 Creating performance comparison plot...")
        create_combined_price_plot()
        
        print("\n" + "="*50)
        print("🎉 All plots created successfully!")
        print("\nGenerated files:")
        print("  📊 stock_prices_subplots.png/pdf - Individual stock charts")
        print("  📈 stock_performance_comparison.png/pdf - Performance comparison")
        
    except Exception as e:
        print(f"❌ Error creating plots: {str(e)}")
        raise

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Clean Stock Cycle Visualization
Creates clear, meaningful visualizations of stock cycles and patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set clean style
plt.style.use('default')
sns.set_palette("husl")

class CleanCycleVisualizer:
    """
    Creates clean, meaningful cycle visualizations.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        
        # Company names
        self.company_names = {
            'AAPL': 'Apple',
            'NVDA': 'NVIDIA', 
            'LLY': 'Eli Lilly',
            'NVO': 'Novo Nordisk',
            'DNA': 'Genentech'
        }
    
    def load_data(self):
        """Load the stock data."""
        filepath = f"{self.data_dir}/all_stocks_combined.csv"
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'], utc=True)
        return data
    
    def create_clean_price_plots(self, data):
        """Create clean price plots with clear patterns."""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Stock Price Patterns and Cycles', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        tickers = data['Ticker'].unique()
        
        # Plot individual stocks
        for i, ticker in enumerate(tickers):
            if i >= 5:  # Only plot 5 stocks
                break
                
            ticker_data = data[data['Ticker'] == ticker].sort_values('Date')
            company_name = self.company_names.get(ticker, ticker)
            
            # Plot price with moving averages
            ax = axes[i]
            ax.plot(ticker_data['Date'], ticker_data['Close'], 
                   linewidth=1.5, alpha=0.8, label='Price')
            
            # Add moving averages
            ticker_data['MA_20'] = ticker_data['Close'].rolling(window=20).mean()
            ticker_data['MA_50'] = ticker_data['Close'].rolling(window=50).mean()
            
            ax.plot(ticker_data['Date'], ticker_data['MA_20'], 
                   '--', alpha=0.7, linewidth=1, color='orange', label='20-day MA')
            ax.plot(ticker_data['Date'], ticker_data['MA_50'], 
                   '--', alpha=0.7, linewidth=1, color='red', label='50-day MA')
            
            # Formatting
            ax.set_title(f'{company_name} ({ticker})', fontsize=12, fontweight='bold')
            ax.set_ylabel('Price ($)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Format x-axis
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
        
        # Remove the last empty subplot
        if len(tickers) < 6:
            axes[5].remove()
        
        plt.tight_layout()
        plt.savefig('clean_stock_prices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_seasonal_analysis(self, data):
        """Create clear seasonal pattern analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Seasonal Patterns Analysis', fontsize=16, fontweight='bold')
        
        # 1. Monthly Performance
        ax1 = axes[0, 0]
        monthly_data = []
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].copy()
            ticker_data['Month'] = ticker_data['Date'].dt.month
            monthly_perf = ticker_data.groupby('Month')['Daily_Return'].mean()
            
            company_name = self.company_names.get(ticker, ticker)
            ax1.plot(monthly_perf.index, monthly_perf.values, 
                    marker='o', label=company_name, linewidth=2)
        
        ax1.set_title('Average Monthly Returns', fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Average Daily Return (%)')
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 2. Day of Week Performance
        ax2 = axes[0, 1]
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].copy()
            ticker_data['DayOfWeek'] = ticker_data['Date'].dt.dayofweek
            dow_perf = ticker_data.groupby('DayOfWeek')['Daily_Return'].mean()
            
            company_name = self.company_names.get(ticker, ticker)
            ax2.plot(dow_perf.index, dow_perf.values, 
                    marker='s', label=company_name, linewidth=2)
        
        ax2.set_title('Average Day-of-Week Returns', fontweight='bold')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Average Daily Return (%)')
        ax2.set_xticks(range(5))
        ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 3. Monthly Volatility
        ax3 = axes[1, 0]
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].copy()
            ticker_data['Month'] = ticker_data['Date'].dt.month
            monthly_vol = ticker_data.groupby('Month')['Daily_Return'].std()
            
            company_name = self.company_names.get(ticker, ticker)
            ax3.plot(monthly_vol.index, monthly_vol.values, 
                    marker='^', label=company_name, linewidth=2)
        
        ax3.set_title('Monthly Volatility', fontweight='bold')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Daily Return Volatility (%)')
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Performance Summary
        ax4 = axes[1, 1]
        
        # Calculate summary statistics
        summary_data = []
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker]
            company_name = self.company_names.get(ticker, ticker)
            
            summary_data.append({
                'Company': company_name,
                'Total Return': ((ticker_data['Close'].iloc[-1] / ticker_data['Close'].iloc[0]) - 1) * 100,
                'Volatility': ticker_data['Daily_Return'].std() * 100,
                'Sharpe Ratio': ticker_data['Daily_Return'].mean() / ticker_data['Daily_Return'].std() * np.sqrt(252)
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create bar chart
        x = np.arange(len(summary_df))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, summary_df['Total Return'], width, 
                       label='Total Return (%)', alpha=0.8)
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x + width/2, summary_df['Volatility'], width, 
                            label='Volatility (%)', alpha=0.8, color='orange')
        
        ax4.set_title('Performance Summary', fontweight='bold')
        ax4.set_xlabel('Company')
        ax4.set_ylabel('Total Return (%)')
        ax4_twin.set_ylabel('Volatility (%)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(summary_df['Company'], rotation=45)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('clean_seasonal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_cycle_patterns(self, data):
        """Create clear cycle pattern analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cycle Pattern Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price with Trend Lines
        ax1 = axes[0, 0]
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].sort_values('Date')
            company_name = self.company_names.get(ticker, ticker)
            
            # Plot price
            ax1.plot(ticker_data['Date'], ticker_data['Close'], 
                    label=company_name, alpha=0.7, linewidth=1)
        
        ax1.set_title('Price Trends Over Time', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling Correlation
        ax2 = axes[0, 1]
        
        # Calculate rolling correlation between stocks
        pivot_data = data.pivot(index='Date', columns='Ticker', values='Close')
        rolling_corr = pivot_data.rolling(window=30).corr().unstack()
        
        # Plot correlation between Apple and NVIDIA as example
        if 'AAPL' in pivot_data.columns and 'NVDA' in pivot_data.columns:
            corr_series = rolling_corr['AAPL']['NVDA'].dropna()
            ax2.plot(corr_series.index, corr_series.values, 
                    linewidth=2, color='blue', alpha=0.8)
            ax2.set_title('Rolling Correlation: Apple vs NVIDIA', fontweight='bold')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Correlation')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 3. Volume Patterns
        ax3 = axes[1, 0]
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].sort_values('Date')
            company_name = self.company_names.get(ticker, ticker)
            
            # Calculate rolling average volume
            ticker_data['Volume_MA'] = ticker_data['Volume'].rolling(window=20).mean()
            
            ax3.plot(ticker_data['Date'], ticker_data['Volume_MA'], 
                    label=company_name, alpha=0.7, linewidth=1)
        
        ax3.set_title('Volume Trends (20-day MA)', fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Average Volume')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')  # Log scale for better visualization
        
        # 4. Return Distribution
        ax4 = axes[1, 1]
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker]
            company_name = self.company_names.get(ticker, ticker)
            
            # Plot histogram of daily returns
            ax4.hist(ticker_data['Daily_Return'].dropna(), 
                    bins=30, alpha=0.6, label=company_name, density=True)
        
        ax4.set_title('Daily Return Distributions', fontweight='bold')
        ax4.set_xlabel('Daily Return (%)')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('clean_cycle_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_trading_signals(self, data):
        """Create clear trading signal analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Trading Signal Analysis', fontsize=16, fontweight='bold')
        
        # Focus on one stock for detailed analysis (Apple)
        ticker = 'AAPL'
        ticker_data = data[data['Ticker'] == ticker].sort_values('Date')
        
        # 1. Price with Moving Averages and Signals
        ax1 = axes[0, 0]
        
        # Calculate moving averages
        ticker_data['MA_10'] = ticker_data['Close'].rolling(window=10).mean()
        ticker_data['MA_30'] = ticker_data['Close'].rolling(window=30).mean()
        
        # Plot price and moving averages
        ax1.plot(ticker_data['Date'], ticker_data['Close'], 
                linewidth=2, label='Price', color='black')
        ax1.plot(ticker_data['Date'], ticker_data['MA_10'], 
                linewidth=1, label='10-day MA', color='blue')
        ax1.plot(ticker_data['Date'], ticker_data['MA_30'], 
                linewidth=1, label='30-day MA', color='red')
        
        # Add buy/sell signals
        ticker_data['Signal'] = np.where(ticker_data['MA_10'] > ticker_data['MA_30'], 1, 0)
        ticker_data['Position'] = ticker_data['Signal'].diff()
        
        # Plot buy signals
        buy_signals = ticker_data[ticker_data['Position'] == 1]
        sell_signals = ticker_data[ticker_data['Position'] == -1]
        
        ax1.scatter(buy_signals['Date'], buy_signals['Close'], 
                  color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        ax1.scatter(sell_signals['Date'], sell_signals['Close'], 
                  color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title(f'{self.company_names.get(ticker, ticker)} - Moving Average Signals', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI Analysis
        ax2 = axes[0, 1]
        
        # Calculate RSI
        delta = ticker_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        ax2.plot(ticker_data['Date'], rsi, linewidth=2, color='purple')
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
        ax2.fill_between(ticker_data['Date'], 30, 70, alpha=0.1, color='gray')
        
        ax2.set_title(f'{self.company_names.get(ticker, ticker)} - RSI Analysis', fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Bollinger Bands
        ax3 = axes[1, 0]
        
        # Calculate Bollinger Bands
        ticker_data['BB_Middle'] = ticker_data['Close'].rolling(window=20).mean()
        bb_std = ticker_data['Close'].rolling(window=20).std()
        ticker_data['BB_Upper'] = ticker_data['BB_Middle'] + (bb_std * 2)
        ticker_data['BB_Lower'] = ticker_data['BB_Middle'] - (bb_std * 2)
        
        ax3.plot(ticker_data['Date'], ticker_data['Close'], 
                linewidth=2, label='Price', color='black')
        ax3.plot(ticker_data['Date'], ticker_data['BB_Upper'], 
                linewidth=1, label='Upper Band', color='red', alpha=0.7)
        ax3.plot(ticker_data['Date'], ticker_data['BB_Lower'], 
                linewidth=1, label='Lower Band', color='green', alpha=0.7)
        ax3.fill_between(ticker_data['Date'], ticker_data['BB_Lower'], ticker_data['BB_Upper'], 
                        alpha=0.1, color='gray')
        
        ax3.set_title(f'{self.company_names.get(ticker, ticker)} - Bollinger Bands', fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Price ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. MACD
        ax4 = axes[1, 1]
        
        # Calculate MACD
        ema_12 = ticker_data['Close'].ewm(span=12).mean()
        ema_26 = ticker_data['Close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        ax4.plot(ticker_data['Date'], macd, linewidth=2, label='MACD', color='blue')
        ax4.plot(ticker_data['Date'], signal, linewidth=2, label='Signal', color='red')
        ax4.bar(ticker_data['Date'], histogram, alpha=0.3, label='Histogram', color='gray')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax4.set_title(f'{self.company_names.get(ticker, ticker)} - MACD', fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('MACD')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('clean_trading_signals.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to create clean visualizations."""
    try:
        print("🎨 Creating Clean Stock Visualizations...")
        print("="*60)
        
        # Initialize visualizer
        visualizer = CleanCycleVisualizer()
        
        # Load data
        print("📊 Loading stock data...")
        data = visualizer.load_data()
        print(f"Loaded data for {data['Ticker'].nunique()} stocks")
        
        # Create clean visualizations
        print("\n📈 Creating clean price plots...")
        visualizer.create_clean_price_plots(data)
        
        print("📅 Creating seasonal analysis...")
        visualizer.create_seasonal_analysis(data)
        
        print("🔄 Creating cycle patterns...")
        visualizer.create_cycle_patterns(data)
        
        print("📊 Creating trading signals...")
        visualizer.create_trading_signals(data)
        
        print("\n" + "="*60)
        print("🎉 CLEAN VISUALIZATIONS COMPLETED!")
        print("="*60)
        print("\n📁 Files created:")
        print("  📊 clean_stock_prices.png - Clear price charts")
        print("  📅 clean_seasonal_analysis.png - Seasonal patterns")
        print("  🔄 clean_cycle_patterns.png - Cycle analysis")
        print("  📊 clean_trading_signals.png - Trading signals")
        
        print("\n✨ These plots are much cleaner and more meaningful!")
        
    except Exception as e:
        print(f"❌ Error creating clean visualizations: {str(e)}")
        raise

if __name__ == "__main__":
    main()

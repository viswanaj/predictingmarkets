#!/usr/bin/env python3
"""
Stock Cycle Analysis
Analyzes cyclical patterns, seasonal trends, and recurring swings in stock prices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StockCycleAnalyzer:
    """
    Analyzes cyclical patterns in stock data.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        
        # Company names for better visualization
        self.company_names = {
            'AAPL': 'Apple Inc.',
            'NVDA': 'NVIDIA Corporation', 
            'LLY': 'Eli Lilly and Company',
            'NVO': 'Novo Nordisk A/S',
            'DNA': 'Genentech (Roche)'
        }
    
    def load_daily_data(self):
        """Load the daily stock data."""
        filepath = f"{self.data_dir}/all_stocks_combined.csv"
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'], utc=True)
        return data
    
    def detect_cycles_fft(self, price_data, min_period=5, max_period=252):
        """
        Detect cycles using Fast Fourier Transform.
        
        Args:
            price_data: Series of prices
            min_period: Minimum cycle period (days)
            max_period: Maximum cycle period (days)
            
        Returns:
            Dictionary with cycle information
        """
        # Remove NaN values
        clean_data = price_data.dropna()
        
        if len(clean_data) < max_period:
            return None
        
        # Calculate returns for better cycle detection
        returns = clean_data.pct_change().dropna()
        
        # Apply FFT
        fft_values = fft(returns.values)
        freqs = fftfreq(len(returns))
        
        # Convert frequencies to periods (in days)
        periods = 1 / np.abs(freqs[1:len(freqs)//2])
        
        # Filter for relevant periods
        valid_mask = (periods >= min_period) & (periods <= max_period)
        valid_periods = periods[valid_mask]
        valid_fft = np.abs(fft_values[1:len(fft_values)//2])[valid_mask]
        
        if len(valid_periods) == 0:
            return None
        
        # Find dominant cycles
        dominant_indices = np.argsort(valid_fft)[-3:]  # Top 3 cycles
        dominant_periods = valid_periods[dominant_indices]
        dominant_powers = valid_fft[dominant_indices]
        
        return {
            'periods': dominant_periods,
            'powers': dominant_powers,
            'all_periods': valid_periods,
            'all_powers': valid_fft
        }
    
    def detect_seasonal_patterns(self, data):
        """Detect seasonal patterns in stock data."""
        seasonal_analysis = {}
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('Date')
            
            # Add time-based features
            ticker_data['Year'] = ticker_data['Date'].dt.year
            ticker_data['Month'] = ticker_data['Date'].dt.month
            ticker_data['DayOfWeek'] = ticker_data['Date'].dt.dayofweek
            ticker_data['DayOfMonth'] = ticker_data['Date'].dt.day
            ticker_data['Quarter'] = ticker_data['Date'].dt.quarter
            
            # Calculate monthly returns
            monthly_returns = ticker_data.groupby(['Year', 'Month'])['Close'].last().pct_change()
            
            # Calculate average monthly performance
            monthly_performance = ticker_data.groupby('Month')['Daily_Return'].mean()
            
            # Calculate day-of-week patterns
            dow_performance = ticker_data.groupby('DayOfWeek')['Daily_Return'].mean()
            
            # Calculate quarterly patterns
            quarterly_performance = ticker_data.groupby('Quarter')['Daily_Return'].mean()
            
            seasonal_analysis[ticker] = {
                'monthly_performance': monthly_performance,
                'dow_performance': dow_performance,
                'quarterly_performance': quarterly_performance,
                'monthly_volatility': ticker_data.groupby('Month')['Daily_Return'].std(),
                'dow_volatility': ticker_data.groupby('DayOfWeek')['Daily_Return'].std()
            }
        
        return seasonal_analysis
    
    def detect_swing_patterns(self, data, window=20):
        """Detect swing patterns (local highs and lows)."""
        swing_analysis = {}
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('Date')
            
            # Calculate rolling highs and lows
            ticker_data['Rolling_High'] = ticker_data['High'].rolling(window=window).max()
            ticker_data['Rolling_Low'] = ticker_data['Low'].rolling(window=window).min()
            
            # Identify swing highs and lows
            ticker_data['Is_Swing_High'] = (
                (ticker_data['High'] == ticker_data['Rolling_High']) & 
                (ticker_data['High'].shift(1) < ticker_data['High']) &
                (ticker_data['High'].shift(-1) < ticker_data['High'])
            )
            
            ticker_data['Is_Swing_Low'] = (
                (ticker_data['Low'] == ticker_data['Rolling_Low']) & 
                (ticker_data['Low'].shift(1) > ticker_data['Low']) &
                (ticker_data['Low'].shift(-1) > ticker_data['Low'])
            )
            
            # Calculate swing durations and magnitudes
            swing_highs = ticker_data[ticker_data['Is_Swing_High']].copy()
            swing_lows = ticker_data[ticker_data['Is_Swing_Low']].copy()
            
            if len(swing_highs) > 1 and len(swing_lows) > 1:
                # Calculate average swing duration
                high_durations = swing_highs['Date'].diff().dt.days.dropna()
                low_durations = swing_lows['Date'].diff().dt.days.dropna()
                
                # Calculate swing magnitudes
                high_magnitudes = swing_highs['High'].pct_change().dropna()
                low_magnitudes = swing_lows['Low'].pct_change().dropna()
                
                swing_analysis[ticker] = {
                    'avg_high_duration': high_durations.mean(),
                    'avg_low_duration': low_durations.mean(),
                    'avg_high_magnitude': high_magnitudes.mean(),
                    'avg_low_magnitude': low_magnitudes.mean(),
                    'swing_high_count': len(swing_highs),
                    'swing_low_count': len(swing_lows),
                    'swing_data': ticker_data
                }
            else:
                swing_analysis[ticker] = None
        
        return swing_analysis
    
    def analyze_momentum_cycles(self, data):
        """Analyze momentum cycles using technical indicators."""
        momentum_analysis = {}
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('Date')
            
            # Calculate momentum indicators
            ticker_data['RSI'] = self.calculate_rsi(ticker_data['Close'])
            ticker_data['MACD'] = self.calculate_macd(ticker_data['Close'])
            ticker_data['MACD_Signal'] = ticker_data['MACD'].ewm(span=9).mean()
            ticker_data['MACD_Histogram'] = ticker_data['MACD'] - ticker_data['MACD_Signal']
            
            # Identify momentum cycles
            ticker_data['RSI_Oversold'] = ticker_data['RSI'] < 30
            ticker_data['RSI_Overbought'] = ticker_data['RSI'] > 70
            ticker_data['MACD_Bullish'] = ticker_data['MACD'] > ticker_data['MACD_Signal']
            ticker_data['MACD_Bearish'] = ticker_data['MACD'] < ticker_data['MACD_Signal']
            
            # Calculate cycle lengths
            rsi_cycles = self.calculate_cycle_lengths(ticker_data['RSI_Oversold'])
            macd_cycles = self.calculate_cycle_lengths(ticker_data['MACD_Bullish'])
            
            momentum_analysis[ticker] = {
                'avg_rsi_cycle': rsi_cycles['avg_cycle_length'] if rsi_cycles else None,
                'avg_macd_cycle': macd_cycles['avg_cycle_length'] if macd_cycles else None,
                'rsi_data': ticker_data[['Date', 'RSI', 'RSI_Oversold', 'RSI_Overbought']],
                'macd_data': ticker_data[['Date', 'MACD', 'MACD_Signal', 'MACD_Bullish', 'MACD_Bearish']]
            }
        
        return momentum_analysis
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def calculate_cycle_lengths(self, signal_series):
        """Calculate average cycle lengths from a boolean signal."""
        if signal_series.sum() < 2:
            return None
        
        # Find where signal changes from False to True
        signal_changes = signal_series.diff() == 1
        cycle_starts = signal_series[signal_changes].index
        
        if len(cycle_starts) < 2:
            return None
        
        # Calculate cycle lengths
        cycle_lengths = []
        for i in range(1, len(cycle_starts)):
            cycle_length = cycle_starts[i] - cycle_starts[i-1]
            cycle_lengths.append(cycle_length)
        
        return {
            'avg_cycle_length': np.mean(cycle_lengths),
            'std_cycle_length': np.std(cycle_lengths),
            'min_cycle_length': np.min(cycle_lengths),
            'max_cycle_length': np.max(cycle_lengths),
            'cycle_lengths': cycle_lengths
        }
    
    def create_cycle_visualizations(self, data, seasonal_analysis, swing_analysis, momentum_analysis):
        """Create comprehensive cycle visualizations."""
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. FFT Cycle Analysis
        ax1 = plt.subplot(4, 2, 1)
        self.plot_fft_cycles(data, ax1)
        
        # 2. Seasonal Patterns
        ax2 = plt.subplot(4, 2, 2)
        self.plot_seasonal_patterns(seasonal_analysis, ax2)
        
        # 3. Day of Week Patterns
        ax3 = plt.subplot(4, 2, 3)
        self.plot_dow_patterns(seasonal_analysis, ax3)
        
        # 4. Monthly Performance Heatmap
        ax4 = plt.subplot(4, 2, 4)
        self.plot_monthly_heatmap(seasonal_analysis, ax4)
        
        # 5. Swing Patterns
        ax5 = plt.subplot(4, 2, 5)
        self.plot_swing_patterns(swing_analysis, ax5)
        
        # 6. Momentum Cycles
        ax6 = plt.subplot(4, 2, 6)
        self.plot_momentum_cycles(momentum_analysis, ax6)
        
        # 7. Cycle Summary
        ax7 = plt.subplot(4, 2, 7)
        self.plot_cycle_summary(data, seasonal_analysis, swing_analysis, momentum_analysis, ax7)
        
        # 8. Price with Cycle Overlay
        ax8 = plt.subplot(4, 2, 8)
        self.plot_price_with_cycles(data, ax8)
        
        plt.tight_layout()
        plt.savefig('stock_cycle_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig('stock_cycle_analysis.pdf', bbox_inches='tight')
        plt.show()
    
    def plot_fft_cycles(self, data, ax):
        """Plot FFT cycle analysis."""
        cycle_periods = []
        cycle_powers = []
        tickers = []
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].sort_values('Date')
            cycle_info = self.detect_cycles_fft(ticker_data['Close'])
            
            if cycle_info:
                cycle_periods.extend(cycle_info['periods'])
                cycle_powers.extend(cycle_info['powers'])
                tickers.extend([ticker] * len(cycle_info['periods']))
        
        if cycle_periods:
            cycle_df = pd.DataFrame({
                'Period': cycle_periods,
                'Power': cycle_powers,
                'Ticker': tickers
            })
            
            for ticker in cycle_df['Ticker'].unique():
                ticker_cycles = cycle_df[cycle_df['Ticker'] == ticker]
                ax.scatter(ticker_cycles['Period'], ticker_cycles['Power'], 
                          label=self.company_names.get(ticker, ticker), s=100, alpha=0.7)
        
        ax.set_xlabel('Cycle Period (Days)')
        ax.set_ylabel('Cycle Power')
        ax.set_title('FFT Cycle Analysis - Dominant Cycles')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_seasonal_patterns(self, seasonal_analysis, ax):
        """Plot seasonal patterns."""
        months = range(1, 13)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for ticker, analysis in seasonal_analysis.items():
            monthly_perf = analysis['monthly_performance']
            ax.plot(months, monthly_perf.values, 
                   marker='o', label=self.company_names.get(ticker, ticker), linewidth=2)
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Monthly Return')
        ax.set_title('Seasonal Patterns - Monthly Performance')
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    def plot_dow_patterns(self, seasonal_analysis, ax):
        """Plot day-of-week patterns."""
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        
        for ticker, analysis in seasonal_analysis.items():
            dow_perf = analysis['dow_performance']
            ax.plot(days, dow_perf.values, 
                   marker='s', label=self.company_names.get(ticker, ticker), linewidth=2)
        
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Average Daily Return')
        ax.set_title('Day-of-Week Patterns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    def plot_monthly_heatmap(self, seasonal_analysis, ax):
        """Plot monthly performance heatmap."""
        # Create a matrix of monthly returns
        monthly_matrix = []
        tickers = list(seasonal_analysis.keys())
        
        for ticker in tickers:
            monthly_perf = seasonal_analysis[ticker]['monthly_performance']
            monthly_matrix.append(monthly_perf.values)
        
        monthly_matrix = np.array(monthly_matrix)
        
        # Create heatmap
        im = ax.imshow(monthly_matrix, cmap='RdYlGn', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(range(len(tickers)))
        ax.set_yticklabels([self.company_names.get(t, t) for t in tickers])
        
        ax.set_title('Monthly Performance Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Average Monthly Return')
    
    def plot_swing_patterns(self, swing_analysis, ax):
        """Plot swing pattern analysis."""
        tickers = []
        avg_durations = []
        avg_magnitudes = []
        
        for ticker, analysis in swing_analysis.items():
            if analysis:
                tickers.append(self.company_names.get(ticker, ticker))
                avg_durations.append(analysis['avg_high_duration'])
                avg_magnitudes.append(analysis['avg_high_magnitude'])
        
        if tickers:
            x = np.arange(len(tickers))
            width = 0.35
            
            ax.bar(x - width/2, avg_durations, width, label='Avg Duration (Days)', alpha=0.8)
            ax2 = ax.twinx()
            ax2.bar(x + width/2, avg_magnitudes, width, label='Avg Magnitude', alpha=0.8, color='orange')
            
            ax.set_xlabel('Company')
            ax.set_ylabel('Average Swing Duration (Days)')
            ax2.set_ylabel('Average Swing Magnitude')
            ax.set_title('Swing Pattern Analysis')
            ax.set_xticks(x)
            ax.set_xticklabels(tickers, rotation=45)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
    
    def plot_momentum_cycles(self, momentum_analysis, ax):
        """Plot momentum cycle analysis."""
        tickers = []
        rsi_cycles = []
        macd_cycles = []
        
        for ticker, analysis in momentum_analysis.items():
            if analysis['avg_rsi_cycle'] and analysis['avg_macd_cycle']:
                tickers.append(self.company_names.get(ticker, ticker))
                rsi_cycles.append(analysis['avg_rsi_cycle'])
                macd_cycles.append(analysis['avg_macd_cycle'])
        
        if tickers:
            x = np.arange(len(tickers))
            width = 0.35
            
            ax.bar(x - width/2, rsi_cycles, width, label='RSI Cycle Length', alpha=0.8)
            ax.bar(x + width/2, macd_cycles, width, label='MACD Cycle Length', alpha=0.8)
            
            ax.set_xlabel('Company')
            ax.set_ylabel('Average Cycle Length (Days)')
            ax.set_title('Momentum Cycle Analysis')
            ax.set_xticks(x)
            ax.set_xticklabels(tickers, rotation=45)
            ax.legend()
    
    def plot_cycle_summary(self, data, seasonal_analysis, swing_analysis, momentum_analysis, ax):
        """Plot cycle summary statistics."""
        summary_data = []
        
        for ticker in data['Ticker'].unique():
            company_name = self.company_names.get(ticker, ticker)
            
            # Get seasonal info
            seasonal_info = seasonal_analysis.get(ticker, {})
            best_month = seasonal_info.get('monthly_performance', pd.Series()).idxmax() if seasonal_info.get('monthly_performance') is not None else 'N/A'
            worst_month = seasonal_info.get('monthly_performance', pd.Series()).idxmin() if seasonal_info.get('monthly_performance') is not None else 'N/A'
            
            # Get swing info
            swing_info = swing_analysis.get(ticker)
            avg_swing_duration = swing_info['avg_high_duration'] if swing_info else 'N/A'
            
            # Get momentum info
            momentum_info = momentum_analysis.get(ticker)
            avg_rsi_cycle = momentum_info['avg_rsi_cycle'] if momentum_info and momentum_info['avg_rsi_cycle'] else 'N/A'
            
            summary_data.append({
                'Company': company_name,
                'Best Month': best_month,
                'Worst Month': worst_month,
                'Avg Swing Duration': avg_swing_duration,
                'Avg RSI Cycle': avg_rsi_cycle
            })
        
        # Create summary table
        ax.axis('tight')
        ax.axis('off')
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            table = ax.table(cellText=summary_df.values,
                           colLabels=summary_df.columns,
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax.set_title('Cycle Analysis Summary', fontsize=14, fontweight='bold')
    
    def plot_price_with_cycles(self, data, ax):
        """Plot price data with cycle overlays."""
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].sort_values('Date')
            
            # Plot price
            ax.plot(ticker_data['Date'], ticker_data['Close'], 
                   label=self.company_names.get(ticker, ticker), alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title('Stock Prices with Cycle Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def generate_cycle_report(self, data, seasonal_analysis, swing_analysis, momentum_analysis):
        """Generate a comprehensive cycle analysis report."""
        report = []
        
        report.append("="*80)
        report.append("STOCK CYCLE ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        for ticker in data['Ticker'].unique():
            company_name = self.company_names.get(ticker, ticker)
            report.append(f"📊 {company_name} ({ticker})")
            report.append("-" * 50)
            
            # Seasonal Analysis
            seasonal_info = seasonal_analysis.get(ticker, {})
            if seasonal_info.get('monthly_performance') is not None:
                monthly_perf = seasonal_info['monthly_performance']
                best_month = monthly_perf.idxmax()
                worst_month = monthly_perf.idxmin()
                best_return = monthly_perf.max()
                worst_return = monthly_perf.min()
                
                report.append(f"📅 Seasonal Patterns:")
                report.append(f"   Best Month: {best_month} ({best_return:.2%})")
                report.append(f"   Worst Month: {worst_month} ({worst_return:.2%})")
                
                # Day of week patterns
                dow_perf = seasonal_info.get('dow_performance', pd.Series())
                if not dow_perf.empty:
                    best_day = dow_perf.idxmax()
                    worst_day = dow_perf.idxmin()
                    report.append(f"   Best Day: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][best_day]} ({dow_perf.iloc[best_day]:.2%})")
                    report.append(f"   Worst Day: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][worst_day]} ({dow_perf.iloc[worst_day]:.2%})")
            
            # Swing Analysis
            swing_info = swing_analysis.get(ticker)
            if swing_info:
                report.append(f"🔄 Swing Patterns:")
                report.append(f"   Average Swing Duration: {swing_info['avg_high_duration']:.1f} days")
                report.append(f"   Average Swing Magnitude: {swing_info['avg_high_magnitude']:.2%}")
                report.append(f"   Total Swing Highs: {swing_info['swing_high_count']}")
                report.append(f"   Total Swing Lows: {swing_info['swing_low_count']}")
            
            # Momentum Analysis
            momentum_info = momentum_analysis.get(ticker)
            if momentum_info and momentum_info['avg_rsi_cycle']:
                report.append(f"⚡ Momentum Cycles:")
                report.append(f"   Average RSI Cycle: {momentum_info['avg_rsi_cycle']:.1f} days")
                if momentum_info['avg_macd_cycle']:
                    report.append(f"   Average MACD Cycle: {momentum_info['avg_macd_cycle']:.1f} days")
            
            # FFT Cycle Analysis
            ticker_data = data[data['Ticker'] == ticker].sort_values('Date')
            cycle_info = self.detect_cycles_fft(ticker_data['Close'])
            if cycle_info:
                report.append(f"🌊 FFT Cycle Analysis:")
                for i, (period, power) in enumerate(zip(cycle_info['periods'], cycle_info['powers'])):
                    report.append(f"   Cycle {i+1}: {period:.1f} days (power: {power:.2f})")
            
            report.append("")
        
        # Overall conclusions
        report.append("🎯 OVERALL CONCLUSIONS")
        report.append("-" * 50)
        
        # Find most cyclical stocks
        cycle_strengths = {}
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].sort_values('Date')
            cycle_info = self.detect_cycles_fft(ticker_data['Close'])
            if cycle_info:
                cycle_strengths[ticker] = np.max(cycle_info['powers'])
        
        if cycle_strengths:
            most_cyclical = max(cycle_strengths, key=cycle_strengths.get)
            least_cyclical = min(cycle_strengths, key=cycle_strengths.get)
            
            report.append(f"Most Cyclical Stock: {self.company_names.get(most_cyclical, most_cyclical)}")
            report.append(f"Least Cyclical Stock: {self.company_names.get(least_cyclical, least_cyclical)}")
        
        report.append("")
        report.append("💡 KEY INSIGHTS:")
        report.append("• All stocks show some degree of cyclical behavior")
        report.append("• Seasonal patterns vary significantly between companies")
        report.append("• Swing patterns can be used for timing entries/exits")
        report.append("• Momentum cycles provide additional trading signals")
        report.append("• FFT analysis reveals hidden periodic patterns")
        
        return "\n".join(report)

def main():
    """Main function to run cycle analysis."""
    try:
        print("🔄 Starting Stock Cycle Analysis...")
        print("="*60)
        
        # Initialize analyzer
        analyzer = StockCycleAnalyzer()
        
        # Load data
        print("📊 Loading stock data...")
        data = analyzer.load_daily_data()
        print(f"Loaded data for {data['Ticker'].nunique()} stocks")
        
        # Perform analyses
        print("\n🔍 Analyzing seasonal patterns...")
        seasonal_analysis = analyzer.detect_seasonal_patterns(data)
        
        print("🔄 Analyzing swing patterns...")
        swing_analysis = analyzer.detect_swing_patterns(data)
        
        print("⚡ Analyzing momentum cycles...")
        momentum_analysis = analyzer.analyze_momentum_cycles(data)
        
        # Create visualizations
        print("\n📈 Creating cycle visualizations...")
        analyzer.create_cycle_visualizations(data, seasonal_analysis, swing_analysis, momentum_analysis)
        
        # Generate report
        print("\n📝 Generating cycle analysis report...")
        report = analyzer.generate_cycle_report(data, seasonal_analysis, swing_analysis, momentum_analysis)
        
        # Save report
        with open('stock_cycle_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print("\n" + "="*60)
        print("🎉 CYCLE ANALYSIS COMPLETED!")
        print("="*60)
        print("\n📁 Files created:")
        print("  📊 stock_cycle_analysis.png/pdf - Comprehensive cycle visualizations")
        print("  📝 stock_cycle_analysis_report.txt - Detailed analysis report")
        
        print("\n" + report)
        
    except Exception as e:
        print(f"❌ Error in cycle analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()

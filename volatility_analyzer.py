#!/usr/bin/env python3
"""
Stock Volatility Analysis
Analyzes volatility across different time horizons: 1 day, 1 week, 1 month, and 1 year.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set clean style
plt.style.use('default')
sns.set_palette("husl")

class VolatilityAnalyzer:
    """
    Analyzes stock volatility across different time horizons.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        
        # Company names
        self.company_names = {
            'AAPL': 'Apple Inc.',
            'NVDA': 'NVIDIA Corporation', 
            'LLY': 'Eli Lilly and Company',
            'NVO': 'Novo Nordisk A/S',
            'DNA': 'Genentech (Roche)'
        }
    
    def load_data(self):
        """Load the stock data."""
        filepath = f"{self.data_dir}/all_stocks_combined.csv"
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'], utc=True)
        return data
    
    def calculate_volatility_metrics(self, data):
        """Calculate volatility metrics for different time horizons."""
        
        volatility_results = {}
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].sort_values('Date')
            company_name = self.company_names.get(ticker, ticker)
            
            # Calculate daily returns
            ticker_data['Daily_Return'] = ticker_data['Close'].pct_change()
            
            # 1. Daily Volatility (using daily returns)
            daily_volatility = ticker_data['Daily_Return'].std() * 100
            
            # 2. Weekly Volatility (using 5-day rolling window)
            ticker_data['Weekly_Return'] = ticker_data['Close'].pct_change(periods=5)
            weekly_volatility = ticker_data['Weekly_Return'].std() * 100
            
            # 3. Monthly Volatility (using 20-day rolling window, approximating 1 month)
            ticker_data['Monthly_Return'] = ticker_data['Close'].pct_change(periods=20)
            monthly_volatility = ticker_data['Monthly_Return'].std() * 100
            
            # 4. Annual Volatility (using 252-day rolling window, approximating 1 year)
            ticker_data['Annual_Return'] = ticker_data['Close'].pct_change(periods=252)
            annual_volatility = ticker_data['Annual_Return'].std() * 100
            
            # Calculate rolling volatility for visualization
            ticker_data['Rolling_Daily_Vol'] = ticker_data['Daily_Return'].rolling(window=20).std() * 100
            ticker_data['Rolling_Weekly_Vol'] = ticker_data['Weekly_Return'].rolling(window=20).std() * 100
            ticker_data['Rolling_Monthly_Vol'] = ticker_data['Monthly_Return'].rolling(window=20).std() * 100
            
            # Calculate volatility percentiles
            daily_vol_percentiles = {
                '25th': np.percentile(ticker_data['Rolling_Daily_Vol'].dropna(), 25),
                '50th': np.percentile(ticker_data['Rolling_Daily_Vol'].dropna(), 50),
                '75th': np.percentile(ticker_data['Rolling_Daily_Vol'].dropna(), 75),
                '95th': np.percentile(ticker_data['Rolling_Daily_Vol'].dropna(), 95)
            }
            
            volatility_results[ticker] = {
                'company_name': company_name,
                'daily_volatility': daily_volatility,
                'weekly_volatility': weekly_volatility,
                'monthly_volatility': monthly_volatility,
                'annual_volatility': annual_volatility,
                'daily_vol_percentiles': daily_vol_percentiles,
                'data': ticker_data
            }
        
        return volatility_results
    
    def create_volatility_visualizations(self, volatility_results):
        """Create comprehensive volatility visualizations."""
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Stock Volatility Analysis Across Time Horizons', fontsize=16, fontweight='bold')
        
        # 1. Volatility Comparison Bar Chart
        ax1 = axes[0, 0]
        
        tickers = list(volatility_results.keys())
        daily_vols = [volatility_results[ticker]['daily_volatility'] for ticker in tickers]
        weekly_vols = [volatility_results[ticker]['weekly_volatility'] for ticker in tickers]
        monthly_vols = [volatility_results[ticker]['monthly_volatility'] for ticker in tickers]
        annual_vols = [volatility_results[ticker]['annual_volatility'] for ticker in tickers]
        
        x = np.arange(len(tickers))
        width = 0.2
        
        ax1.bar(x - 1.5*width, daily_vols, width, label='Daily', alpha=0.8)
        ax1.bar(x - 0.5*width, weekly_vols, width, label='Weekly', alpha=0.8)
        ax1.bar(x + 0.5*width, monthly_vols, width, label='Monthly', alpha=0.8)
        ax1.bar(x + 1.5*width, annual_vols, width, label='Annual', alpha=0.8)
        
        ax1.set_title('Volatility Comparison Across Time Horizons', fontweight='bold')
        ax1.set_xlabel('Company')
        ax1.set_ylabel('Volatility (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([volatility_results[ticker]['company_name'] for ticker in tickers], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling Daily Volatility
        ax2 = axes[0, 1]
        
        for ticker in tickers:
            data = volatility_results[ticker]['data']
            company_name = volatility_results[ticker]['company_name']
            ax2.plot(data['Date'], data['Rolling_Daily_Vol'], 
                    label=company_name, alpha=0.7, linewidth=1.5)
        
        ax2.set_title('Rolling Daily Volatility (20-day window)', fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volatility (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Volatility Heatmap
        ax3 = axes[0, 2]
        
        volatility_matrix = []
        time_horizons = ['Daily', 'Weekly', 'Monthly', 'Annual']
        
        for ticker in tickers:
            row = [
                volatility_results[ticker]['daily_volatility'],
                volatility_results[ticker]['weekly_volatility'],
                volatility_results[ticker]['monthly_volatility'],
                volatility_results[ticker]['annual_volatility']
            ]
            volatility_matrix.append(row)
        
        volatility_matrix = np.array(volatility_matrix)
        
        im = ax3.imshow(volatility_matrix, cmap='YlOrRd', aspect='auto')
        ax3.set_xticks(range(len(time_horizons)))
        ax3.set_xticklabels(time_horizons)
        ax3.set_yticks(range(len(tickers)))
        ax3.set_yticklabels([volatility_results[ticker]['company_name'] for ticker in tickers])
        ax3.set_title('Volatility Heatmap', fontweight='bold')
        
        # Add text annotations
        for i in range(len(tickers)):
            for j in range(len(time_horizons)):
                text = ax3.text(j, i, f'{volatility_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax3, label='Volatility (%)')
        
        # 4. Volatility Distribution
        ax4 = axes[1, 0]
        
        for ticker in tickers:
            data = volatility_results[ticker]['data']
            company_name = volatility_results[ticker]['company_name']
            ax4.hist(data['Rolling_Daily_Vol'].dropna(), bins=30, alpha=0.6, 
                    label=company_name, density=True)
        
        ax4.set_title('Daily Volatility Distribution', fontweight='bold')
        ax4.set_xlabel('Volatility (%)')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Volatility vs Return Scatter
        ax5 = axes[1, 1]
        
        for ticker in tickers:
            data = volatility_results[ticker]['data']
            company_name = volatility_results[ticker]['company_name']
            
            # Calculate rolling returns
            data['Rolling_Return'] = data['Close'].pct_change(periods=20)
            
            ax5.scatter(data['Rolling_Daily_Vol'], data['Rolling_Return'] * 100,
                       label=company_name, alpha=0.6, s=20)
        
        ax5.set_title('Volatility vs Return Relationship', fontweight='bold')
        ax5.set_xlabel('Daily Volatility (%)')
        ax5.set_ylabel('20-day Return (%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 6. Volatility Summary Table
        ax6 = axes[1, 2]
        
        # Create summary table
        summary_data = []
        for ticker in tickers:
            summary_data.append([
                volatility_results[ticker]['company_name'],
                f"{volatility_results[ticker]['daily_volatility']:.2f}",
                f"{volatility_results[ticker]['weekly_volatility']:.2f}",
                f"{volatility_results[ticker]['monthly_volatility']:.2f}",
                f"{volatility_results[ticker]['annual_volatility']:.2f}"
            ])
        
        ax6.axis('tight')
        ax6.axis('off')
        
        table = ax6.table(cellText=summary_data,
                         colLabels=['Company', 'Daily', 'Weekly', 'Monthly', 'Annual'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax6.set_title('Volatility Summary (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('volatility_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig('volatility_analysis.pdf', bbox_inches='tight')
        plt.show()
    
    def generate_volatility_report(self, volatility_results):
        """Generate a comprehensive volatility analysis report."""
        
        report = []
        report.append("="*80)
        report.append("STOCK VOLATILITY ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        # Overall summary
        report.append("📊 VOLATILITY SUMMARY")
        report.append("-" * 50)
        
        # Rank stocks by volatility
        daily_vols = [(ticker, volatility_results[ticker]['daily_volatility']) 
                     for ticker in volatility_results.keys()]
        daily_vols.sort(key=lambda x: x[1], reverse=True)
        
        report.append("Ranking by Daily Volatility (Most to Least Volatile):")
        for i, (ticker, vol) in enumerate(daily_vols, 1):
            company_name = volatility_results[ticker]['company_name']
            report.append(f"  {i}. {company_name} ({ticker}): {vol:.2f}%")
        
        report.append("")
        
        # Detailed analysis for each stock
        for ticker in volatility_results.keys():
            company_name = volatility_results[ticker]['company_name']
            report.append(f"📈 {company_name} ({ticker})")
            report.append("-" * 50)
            
            # Volatility metrics
            report.append("Volatility Metrics:")
            report.append(f"  Daily Volatility: {volatility_results[ticker]['daily_volatility']:.2f}%")
            report.append(f"  Weekly Volatility: {volatility_results[ticker]['weekly_volatility']:.2f}%")
            report.append(f"  Monthly Volatility: {volatility_results[ticker]['monthly_volatility']:.2f}%")
            report.append(f"  Annual Volatility: {volatility_results[ticker]['annual_volatility']:.2f}%")
            
            # Volatility percentiles
            percentiles = volatility_results[ticker]['daily_vol_percentiles']
            report.append("")
            report.append("Daily Volatility Percentiles:")
            report.append(f"  25th Percentile: {percentiles['25th']:.2f}%")
            report.append(f"  50th Percentile (Median): {percentiles['50th']:.2f}%")
            report.append(f"  75th Percentile: {percentiles['75th']:.2f}%")
            report.append(f"  95th Percentile: {percentiles['95th']:.2f}%")
            
            # Risk assessment
            daily_vol = volatility_results[ticker]['daily_volatility']
            if daily_vol > 3.0:
                risk_level = "HIGH"
            elif daily_vol > 2.0:
                risk_level = "MEDIUM-HIGH"
            elif daily_vol > 1.5:
                risk_level = "MEDIUM"
            elif daily_vol > 1.0:
                risk_level = "LOW-MEDIUM"
            else:
                risk_level = "LOW"
            
            report.append("")
            report.append(f"Risk Assessment: {risk_level}")
            
            report.append("")
        
        # Cross-time horizon analysis
        report.append("🔄 CROSS-TIME HORIZON ANALYSIS")
        report.append("-" * 50)
        
        for ticker in volatility_results.keys():
            company_name = volatility_results[ticker]['company_name']
            daily_vol = volatility_results[ticker]['daily_volatility']
            weekly_vol = volatility_results[ticker]['weekly_volatility']
            monthly_vol = volatility_results[ticker]['monthly_volatility']
            annual_vol = volatility_results[ticker]['annual_volatility']
            
            report.append(f"{company_name} ({ticker}):")
            
            # Check if volatility scales properly (should decrease with longer time horizons)
            if weekly_vol < daily_vol * np.sqrt(5):  # Weekly should be ~sqrt(5) * daily
                report.append(f"  ✓ Weekly volatility scales properly")
            else:
                report.append(f"  ⚠ Weekly volatility higher than expected")
            
            if monthly_vol < daily_vol * np.sqrt(20):  # Monthly should be ~sqrt(20) * daily
                report.append(f"  ✓ Monthly volatility scales properly")
            else:
                report.append(f"  ⚠ Monthly volatility higher than expected")
            
            if annual_vol < daily_vol * np.sqrt(252):  # Annual should be ~sqrt(252) * daily
                report.append(f"  ✓ Annual volatility scales properly")
            else:
                report.append(f"  ⚠ Annual volatility higher than expected")
            
            report.append("")
        
        # Key insights
        report.append("💡 KEY INSIGHTS")
        report.append("-" * 50)
        
        most_volatile = max(volatility_results.keys(), 
                          key=lambda x: volatility_results[x]['daily_volatility'])
        least_volatile = min(volatility_results.keys(), 
                           key=lambda x: volatility_results[x]['daily_volatility'])
        
        report.append(f"• Most volatile stock: {volatility_results[most_volatile]['company_name']} "
                    f"({volatility_results[most_volatile]['daily_volatility']:.2f}%)")
        report.append(f"• Least volatile stock: {volatility_results[least_volatile]['company_name']} "
                    f"({volatility_results[least_volatile]['daily_volatility']:.2f}%)")
        
        # Calculate average volatility
        avg_daily_vol = np.mean([volatility_results[ticker]['daily_volatility'] 
                               for ticker in volatility_results.keys()])
        report.append(f"• Average daily volatility across all stocks: {avg_daily_vol:.2f}%")
        
        report.append("")
        report.append("📊 TRADING IMPLICATIONS:")
        report.append("• Higher volatility = Higher risk but potentially higher returns")
        report.append("• Daily volatility useful for day trading risk management")
        report.append("• Weekly/Monthly volatility important for swing trading")
        report.append("• Annual volatility crucial for long-term portfolio allocation")
        report.append("• Volatility clustering often occurs - high vol periods followed by high vol")
        
        return "\n".join(report)

def main():
    """Main function to run volatility analysis."""
    try:
        print("📊 Starting Stock Volatility Analysis...")
        print("="*60)
        
        # Initialize analyzer
        analyzer = VolatilityAnalyzer()
        
        # Load data
        print("📈 Loading stock data...")
        data = analyzer.load_data()
        print(f"Loaded data for {data['Ticker'].nunique()} stocks")
        
        # Calculate volatility metrics
        print("🔍 Calculating volatility metrics...")
        volatility_results = analyzer.calculate_volatility_metrics(data)
        
        # Create visualizations
        print("📊 Creating volatility visualizations...")
        analyzer.create_volatility_visualizations(volatility_results)
        
        # Generate report
        print("📝 Generating volatility report...")
        report = analyzer.generate_volatility_report(volatility_results)
        
        # Save report
        with open('volatility_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print("\n" + "="*60)
        print("🎉 VOLATILITY ANALYSIS COMPLETED!")
        print("="*60)
        print("\n📁 Files created:")
        print("  📊 volatility_analysis.png/pdf - Comprehensive volatility visualizations")
        print("  📝 volatility_analysis_report.txt - Detailed volatility analysis report")
        
        print("\n" + report)
        
    except Exception as e:
        print(f"❌ Error in volatility analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()

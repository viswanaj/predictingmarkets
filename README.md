# Stock Market Data Downloader & Analyzer

A comprehensive Python toolkit for downloading, processing, and analyzing stock market data in indexable formats. This project focuses on pharmaceutical and technology stocks: **Novo Nordisk**, **Eli Lilly**, **Apple**, **NVIDIA**, and **Genentech**.

## 🚀 Features

- **Multi-format Data Export**: CSV, JSON, and Parquet formats
- **Comprehensive Data**: OHLCV data with technical indicators
- **Error Handling**: Robust error handling and logging
- **Technical Analysis**: 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Portfolio Analysis**: Correlation analysis and portfolio metrics
- **Outlier Detection**: Multiple methods for identifying unusual price movements
- **Indexable Format**: Optimized for database indexing and analysis

## 📦 Installation

1. **Clone or download this repository**
2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Quick Start

### Option 1: Run Everything at Once
```bash
python quick_example.py
```

### Option 2: Step-by-Step Usage

#### 1. Download Stock Data
```python
from stock_data_downloader import StockDataDownloader

# Initialize downloader
downloader = StockDataDownloader()

# Download data for all configured stocks
downloader.download_and_save_all(period="2y", interval="1d")
```

#### 2. Process and Analyze Data
```python
from stock_data_processor import StockDataProcessor

# Initialize processor
processor = StockDataProcessor()

# Load data
data = processor.load_combined_data()

# Calculate technical indicators
data_with_indicators = processor.calculate_technical_indicators(data)

# Create price index
price_index = processor.create_price_index(data)

# Calculate correlations
correlation_matrix = processor.calculate_correlation_matrix(data)
```

## 📊 Data Structure

### Stock Data Columns
- `Date`: Trading date
- `Open`, `High`, `Low`, `Close`: OHLC prices
- `Volume`: Trading volume
- `Ticker`: Stock symbol
- `Daily_Return`: Daily percentage return
- `Price_Change`: Absolute price change
- `Price_Change_Pct`: Percentage price change

### Technical Indicators Added
- **Moving Averages**: MA_5, MA_10, MA_20, MA_50, MA_200
- **Exponential Moving Averages**: EMA_12, EMA_26
- **MACD**: MACD, MACD_Signal, MACD_Histogram
- **RSI**: Relative Strength Index
- **Bollinger Bands**: BB_Upper, BB_Lower, BB_Width, BB_Position
- **Stochastic Oscillator**: Stoch_K, Stoch_D
- **ATR**: Average True Range
- **Volume Indicators**: Volume_MA, Volume_Ratio

## 📁 Output Files

### Data Directory (`data/`)
- `NVO_Novo_Nordisk_data.csv` - Individual stock data
- `LLY_Eli_Lilly_data.csv` - Individual stock data
- `AAPL_Apple_data.csv` - Individual stock data
- `NVDA_NVIDIA_data.csv` - Individual stock data
- `DNA_Genentech_data.csv` - Individual stock data
- `all_stocks_combined.csv` - Combined data from all stocks
- `all_stocks_combined.json` - JSON format
- `all_stocks_combined.parquet` - Parquet format (efficient for large datasets)
- `stock_summary_stats.csv` - Summary statistics
- `download_metadata.json` - Download metadata

### Analysis Directory (`analysis/`)
- `data_with_indicators.csv` - Data with technical indicators
- `price_index.csv` - Price indices (base = 100)
- `correlation_matrix.csv` - Stock correlation matrix
- `data_with_outliers.csv` - Data with outlier flags
- `portfolio_analysis.json` - Portfolio metrics and analysis

## 🔧 Configuration

### Adding New Stocks
Edit the `stock_tickers` dictionary in `stock_data_downloader.py`:

```python
self.stock_tickers = {
    "Novo Nordisk": "NVO",
    "Eli Lilly": "LLY", 
    "Apple": "AAPL",
    "NVIDIA": "NVDA",
    "Genentech": "DNA",
    "Your New Stock": "TICKER"  # Add here
}
```

### Changing Data Periods
```python
# Available periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
downloader.download_and_save_all(period="5y", interval="1d")
```

### Available Intervals
- `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m` (intraday)
- `1h` (hourly)
- `1d` (daily)
- `5d`, `1wk`, `1mo`, `3mo` (longer periods)

## 📈 Usage Examples

### Basic Data Loading
```python
import pandas as pd

# Load combined data
data = pd.read_csv('data/all_stocks_combined.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Filter for specific stock
apple_data = data[data['Ticker'] == 'AAPL']
```

### Technical Analysis
```python
from stock_data_processor import StockDataProcessor

processor = StockDataProcessor()
data = processor.load_combined_data()

# Calculate all technical indicators
data_with_indicators = processor.calculate_technical_indicators(data)

# Save enhanced data
data_with_indicators.to_csv('enhanced_stock_data.csv', index=False)
```

### Portfolio Analysis
```python
# Create equal-weight portfolio
portfolio_analysis = processor.create_portfolio_analysis(data)

# Create custom portfolio weights
custom_weights = {
    'AAPL': 0.3,
    'NVDA': 0.3, 
    'LLY': 0.2,
    'NVO': 0.15,
    'DNA': 0.05
}
custom_portfolio = processor.create_portfolio_analysis(data, weights=custom_weights)
```

### Outlier Detection
```python
# Detect outliers using different methods
data_iqr = processor.detect_outliers(data, method='iqr', threshold=1.5)
data_zscore = processor.detect_outliers(data, method='zscore', threshold=2.0)
data_modified_zscore = processor.detect_outliers(data, method='modified_zscore', threshold=3.5)
```

## 🗄️ Database Integration

### SQLite Example
```python
import sqlite3
import pandas as pd

# Load data
data = pd.read_csv('data/all_stocks_combined.csv')

# Create database connection
conn = sqlite3.connect('stock_data.db')

# Create table with proper indexing
data.to_sql('stock_data', conn, if_exists='replace', index=False)

# Create indexes for efficient querying
conn.execute('CREATE INDEX idx_date ON stock_data(Date)')
conn.execute('CREATE INDEX idx_ticker ON stock_data(Ticker)')
conn.execute('CREATE INDEX idx_date_ticker ON stock_data(Date, Ticker)')

conn.close()
```

### PostgreSQL Example
```python
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

# Load data
data = pd.read_csv('data/all_stocks_combined.csv')

# Create database connection
engine = create_engine('postgresql://user:password@localhost/stockdb')

# Upload data
data.to_sql('stock_data', engine, if_exists='replace', index=False)

# Create indexes
with engine.connect() as conn:
    conn.execute('CREATE INDEX idx_date ON stock_data(date)')
    conn.execute('CREATE INDEX idx_ticker ON stock_data(ticker)')
```

## 🔍 Data Quality & Validation

The system includes comprehensive data validation:

- **Missing Data Detection**: Identifies gaps in data
- **Outlier Detection**: Multiple methods (IQR, Z-score, Modified Z-score)
- **Data Consistency Checks**: Validates OHLC relationships
- **Volume Validation**: Checks for unusual volume patterns
- **Date Validation**: Ensures proper date sequences

## 📝 Logging

All operations are logged to `stock_downloader.log` with timestamps and detailed information about:
- Download progress
- Data processing steps
- Error conditions
- File operations

## ⚠️ Important Notes

1. **Rate Limiting**: Yahoo Finance has rate limits. The script includes delays to respect these limits.

2. **Data Availability**: Some stocks may have limited historical data or trading restrictions.

3. **Market Hours**: Data is typically available after market close for the previous trading day.

4. **Holidays**: Stock markets are closed on certain holidays, which will be reflected in the data.

5. **Stock Splits/Dividends**: The data includes adjusted prices that account for stock splits and dividends.

## 🤝 Contributing

Feel free to contribute by:
- Adding new technical indicators
- Improving error handling
- Adding support for additional data sources
- Enhancing the analysis capabilities

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **"No data found for ticker"**
   - Check if the ticker symbol is correct
   - Verify the stock is publicly traded
   - Try a different time period

2. **"Rate limit exceeded"**
   - Wait a few minutes before retrying
   - Reduce the number of stocks being downloaded simultaneously

3. **"File not found" errors**
   - Ensure you've run the downloader first
   - Check that the data directory exists

4. **Memory issues with large datasets**
   - Use Parquet format for better memory efficiency
   - Process data in chunks for very large datasets

### Getting Help

If you encounter issues:
1. Check the log file (`stock_downloader.log`)
2. Verify your internet connection
3. Ensure all required packages are installed
4. Check that the stock tickers are valid

---

**Happy analyzing! 📊📈**

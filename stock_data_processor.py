#!/usr/bin/env python3
"""
Stock Data Processor
Utilities for processing, analyzing, and indexing downloaded stock data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class StockDataProcessor:
    """
    A class to process and analyze downloaded stock data.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the processor.
        
        Args:
            data_dir: Directory containing stock data files
        """
        self.data_dir = data_dir
        
    def load_combined_data(self, filename: str = "all_stocks_combined.csv") -> pd.DataFrame:
        """
        Load the combined stock data from CSV.
        
        Args:
            filename: Name of the combined data file
            
        Returns:
            DataFrame with stock data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'])
        
        logger.info(f"Loaded {len(data)} records from {filename}")
        return data
    
    def load_individual_stock(self, ticker: str) -> pd.DataFrame:
        """
        Load data for a specific stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with stock data
        """
        # Try to find the file for this ticker
        files = os.listdir(self.data_dir)
        ticker_file = None
        
        for file in files:
            if file.startswith(ticker) and file.endswith('.csv'):
                ticker_file = file
                break
        
        if not ticker_file:
            raise FileNotFoundError(f"No data file found for ticker: {ticker}")
        
        filepath = os.path.join(self.data_dir, ticker_file)
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'])
        
        logger.info(f"Loaded {len(data)} records for {ticker}")
        return data
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for stock data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        df = data.copy()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        logger.info("Technical indicators calculated successfully")
        return df
    
    def create_price_index(self, data: pd.DataFrame, base_date: Optional[str] = None) -> pd.DataFrame:
        """
        Create a price index starting from 100 at a base date.
        
        Args:
            data: DataFrame with stock data
            base_date: Base date for index (if None, uses first date)
            
        Returns:
            DataFrame with price index
        """
        df = data.copy()
        
        if base_date is None:
            base_date = df['Date'].min()
        else:
            base_date = pd.to_datetime(base_date)
        
        # Filter data from base date onwards
        df = df[df['Date'] >= base_date].copy()
        
        # Calculate price index for each ticker
        indexed_data = []
        
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].copy()
            
            # Get base price
            base_price = ticker_data[ticker_data['Date'] == base_date]['Close'].iloc[0]
            
            # Calculate index
            ticker_data['Price_Index'] = (ticker_data['Close'] / base_price) * 100
            
            indexed_data.append(ticker_data)
        
        result = pd.concat(indexed_data, ignore_index=True)
        result = result.sort_values(['Date', 'Ticker'])
        
        logger.info(f"Price index created with base date: {base_date}")
        return result
    
    def calculate_correlation_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix between stock returns.
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            Correlation matrix DataFrame
        """
        # Pivot data to have dates as index and tickers as columns
        pivot_data = data.pivot(index='Date', columns='Ticker', values='Daily_Return')
        
        # Calculate correlation matrix
        correlation_matrix = pivot_data.corr()
        
        logger.info("Correlation matrix calculated")
        return correlation_matrix
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers in stock data.
        
        Args:
            data: DataFrame with stock data
            method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier flags
        """
        df = data.copy()
        
        if method == 'iqr':
            # Interquartile Range method
            Q1 = df['Daily_Return'].quantile(0.25)
            Q3 = df['Daily_Return'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df['Is_Outlier'] = (df['Daily_Return'] < lower_bound) | (df['Daily_Return'] > upper_bound)
            
        elif method == 'zscore':
            # Z-score method
            mean_return = df['Daily_Return'].mean()
            std_return = df['Daily_Return'].std()
            df['Z_Score'] = (df['Daily_Return'] - mean_return) / std_return
            df['Is_Outlier'] = np.abs(df['Z_Score']) > threshold
            
        elif method == 'modified_zscore':
            # Modified Z-score using median absolute deviation
            median_return = df['Daily_Return'].median()
            mad = np.median(np.abs(df['Daily_Return'] - median_return))
            df['Modified_Z_Score'] = 0.6745 * (df['Daily_Return'] - median_return) / mad
            df['Is_Outlier'] = np.abs(df['Modified_Z_Score']) > threshold
        
        outlier_count = df['Is_Outlier'].sum()
        logger.info(f"Detected {outlier_count} outliers using {method} method")
        
        return df
    
    def create_portfolio_analysis(self, data: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        Create portfolio analysis from stock data.
        
        Args:
            data: DataFrame with stock data
            weights: Portfolio weights for each ticker (if None, equal weights)
            
        Returns:
            Dictionary with portfolio analysis results
        """
        tickers = data['Ticker'].unique()
        
        if weights is None:
            # Equal weights
            weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
        
        # Calculate daily returns for each stock
        returns_data = data.pivot(index='Date', columns='Ticker', values='Daily_Return')
        
        # Calculate portfolio returns
        portfolio_returns = (returns_data * pd.Series(weights)).sum(axis=1)
        
        # Calculate portfolio metrics
        portfolio_stats = {
            'total_return': (1 + portfolio_returns).prod() - 1,
            'annualized_return': (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252),
            'max_drawdown': (portfolio_returns.cumsum() - portfolio_returns.cumsum().expanding().max()).min(),
            'weights': weights
        }
        
        logger.info("Portfolio analysis completed")
        return portfolio_stats
    
    def export_analysis_results(self, data: pd.DataFrame, output_dir: str = "analysis"):
        """
        Export comprehensive analysis results.
        
        Args:
            data: DataFrame with stock data
            output_dir: Directory to save analysis results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calculate technical indicators
        data_with_indicators = self.calculate_technical_indicators(data)
        
        # Create price index
        price_index = self.create_price_index(data)
        
        # Calculate correlation matrix
        correlation_matrix = self.calculate_correlation_matrix(data)
        
        # Detect outliers
        data_with_outliers = self.detect_outliers(data)
        
        # Create portfolio analysis
        portfolio_analysis = self.create_portfolio_analysis(data)
        
        # Save results
        data_with_indicators.to_csv(os.path.join(output_dir, "data_with_indicators.csv"), index=False)
        price_index.to_csv(os.path.join(output_dir, "price_index.csv"), index=False)
        correlation_matrix.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
        data_with_outliers.to_csv(os.path.join(output_dir, "data_with_outliers.csv"), index=False)
        
        # Save portfolio analysis as JSON
        with open(os.path.join(output_dir, "portfolio_analysis.json"), 'w') as f:
            json.dump(portfolio_analysis, f, indent=2, default=str)
        
        logger.info(f"Analysis results exported to {output_dir}/")
        
        return {
            'data_with_indicators': data_with_indicators,
            'price_index': price_index,
            'correlation_matrix': correlation_matrix,
            'data_with_outliers': data_with_outliers,
            'portfolio_analysis': portfolio_analysis
        }

def main():
    """Main function to run the stock data processor."""
    try:
        # Initialize processor
        processor = StockDataProcessor()
        
        # Load combined data
        data = processor.load_combined_data()
        
        print(f"Loaded data for {data['Ticker'].nunique()} stocks")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"Total records: {len(data)}")
        
        # Export comprehensive analysis
        results = processor.export_analysis_results(data)
        
        print("\n" + "="*60)
        print("STOCK DATA ANALYSIS COMPLETED!")
        print("="*60)
        print("Analysis files created in 'analysis/' directory:")
        print("  - data_with_indicators.csv: Technical indicators")
        print("  - price_index.csv: Price indices")
        print("  - correlation_matrix.csv: Stock correlations")
        print("  - data_with_outliers.csv: Outlier detection")
        print("  - portfolio_analysis.json: Portfolio metrics")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()

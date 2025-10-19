#!/usr/bin/env python3
"""
LSTM Stock Price Prediction System
Trains LSTM models with different neuron configurations to predict 2025 stock prices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class LSTMStockPredictor:
    """
    LSTM-based stock price prediction system.
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
        
        # Model configurations to test
        self.model_configs = {
            'LSTM_2': {'neurons': 2, 'color': 'red'},
            'LSTM_4': {'neurons': 4, 'color': 'orange'},
            'LSTM_8': {'neurons': 8, 'color': 'yellow'},
            'LSTM_16': {'neurons': 16, 'color': 'green'},
            'LSTM_32': {'neurons': 32, 'color': 'blue'},
            'LSTM_64': {'neurons': 64, 'color': 'purple'}
        }
        
        # Training parameters
        self.lookback_days = 60  # Use 60 days to predict next day
        self.test_size = 0.2
        self.random_state = 42
        
    def load_and_prepare_data(self):
        """Load and prepare stock data for LSTM training."""
        
        # Load the main stock data
        filepath = f"{self.data_dir}/all_stocks_combined.csv"
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'], utc=True)
        
        print("📊 Loading and preparing stock data...")
        print(f"Total records: {len(data):,}")
        print(f"Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
        
        # Split data into training (2023-2024) and testing (2025)
        train_data = data[data['Date'].dt.year <= 2024].copy()
        test_data = data[data['Date'].dt.year >= 2025].copy()
        
        print(f"Training data (2023-2024): {len(train_data):,} records")
        print(f"Testing data (2025): {len(test_data):,} records")
        
        return train_data, test_data
    
    def prepare_sequences(self, data, scaler=None, fit_scaler=True):
        """Prepare sequences for LSTM training."""
        
        # Use Close price as the main feature
        prices = data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        if fit_scaler:
            scaler = MinMaxScaler()
            scaled_prices = scaler.fit_transform(prices)
        else:
            scaled_prices = scaler.transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback_days, len(scaled_prices)):
            X.append(scaled_prices[i-self.lookback_days:i, 0])
            y.append(scaled_prices[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM (samples, timesteps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X, y, scaler
    
    def create_lstm_model(self, neurons, input_shape):
        """Create LSTM model with specified number of neurons."""
        
        model = Sequential([
            LSTM(neurons, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(neurons//2, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_models_for_company(self, ticker, train_data, test_data):
        """Train all LSTM models for a specific company."""
        
        company_name = self.company_names.get(ticker, ticker)
        print(f"\n🤖 Training LSTM models for {company_name} ({ticker})...")
        
        # Prepare training data
        company_train_data = train_data[train_data['Ticker'] == ticker].sort_values('Date')
        company_test_data = test_data[test_data['Ticker'] == ticker].sort_values('Date')
        
        if len(company_train_data) < self.lookback_days + 10:
            print(f"⚠️ Insufficient training data for {ticker}")
            return None
        
        # Prepare sequences
        X_train, y_train, scaler = self.prepare_sequences(company_train_data, fit_scaler=True)
        
        if len(company_test_data) > 0:
            X_test, y_test, _ = self.prepare_sequences(company_test_data, scaler=scaler, fit_scaler=False)
        else:
            X_test, y_test = None, None
        
        # Train models
        trained_models = {}
        predictions = {}
        
        for model_name, config in self.model_configs.items():
            neurons = config['neurons']
            print(f"  Training {model_name} ({neurons} neurons)...")
            
            # Create model
            model = self.create_lstm_model(neurons, (X_train.shape[1], 1))
            
            # Train model
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Make predictions
            train_pred = model.predict(X_train, verbose=0)
            train_pred = scaler.inverse_transform(train_pred)
            y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
            
            if X_test is not None:
                test_pred = model.predict(X_test, verbose=0)
                test_pred = scaler.inverse_transform(test_pred)
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            else:
                test_pred = None
                y_test_actual = None
            
            # Store results
            trained_models[model_name] = {
                'model': model,
                'scaler': scaler,
                'history': history
            }
            
            predictions[model_name] = {
                'train_pred': train_pred,
                'train_actual': y_train_actual,
                'test_pred': test_pred,
                'test_actual': y_test_actual,
                'train_dates': company_train_data['Date'].iloc[self.lookback_days:].values,
                'test_dates': company_test_data['Date'].iloc[self.lookback_days:].values if len(company_test_data) > 0 else None
            }
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train_actual, train_pred)
            train_mae = mean_absolute_error(y_train_actual, train_pred)
            
            print(f"    Training MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
            
            if test_pred is not None:
                test_mse = mean_squared_error(y_test_actual, test_pred)
                test_mae = mean_absolute_error(y_test_actual, test_pred)
                print(f"    Testing MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")
        
        return trained_models, predictions
    
    def create_prediction_plots(self, ticker, predictions, actual_2025_data):
        """Create prediction plots for a specific company."""
        
        company_name = self.company_names.get(ticker, ticker)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'LSTM Predictions vs Actual Prices - {company_name} ({ticker})', 
                    fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        # Plot each model
        for i, (model_name, config) in enumerate(self.model_configs.items()):
            ax = axes[i]
            
            pred_data = predictions[model_name]
            color = config['color']
            
            # Plot training predictions
            train_dates = pred_data['train_dates']
            train_actual = pred_data['train_actual'].flatten()
            train_pred = pred_data['train_pred'].flatten()
            
            ax.plot(train_dates, train_actual, 'b-', alpha=0.7, label='Training Actual', linewidth=1)
            ax.plot(train_dates, train_pred, color=color, alpha=0.7, label='Training Predicted', linewidth=1)
            
            # Plot test predictions (2025)
            if pred_data['test_pred'] is not None:
                test_dates = pred_data['test_dates']
                test_actual = pred_data['test_actual'].flatten()
                test_pred = pred_data['test_pred'].flatten()
                
                ax.plot(test_dates, test_actual, 'g-', alpha=0.8, label='2025 Actual', linewidth=2)
                ax.plot(test_dates, test_pred, color=color, alpha=0.8, label='2025 Predicted', linewidth=2)
            
            # Formatting
            ax.set_title(f'{model_name} ({config["neurons"]} neurons)', fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'lstm_predictions_{ticker}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'lstm_predictions_{ticker}.pdf', bbox_inches='tight')
        plt.show()
    
    def create_comparison_plot(self, all_predictions, actual_2025_data):
        """Create a comparison plot showing all models for all companies."""
        
        fig, axes = plt.subplots(5, 1, figsize=(20, 25))
        fig.suptitle('LSTM Model Comparison - All Companies', fontsize=20, fontweight='bold')
        
        for i, ticker in enumerate(self.company_names.keys()):
            ax = axes[i]
            company_name = self.company_names[ticker]
            
            if ticker not in all_predictions:
                continue
            
            predictions = all_predictions[ticker]
            
            # Plot actual 2025 data
            company_2025_data = actual_2025_data[actual_2025_data['Ticker'] == ticker].sort_values('Date')
            if len(company_2025_data) > 0:
                ax.plot(company_2025_data['Date'], company_2025_data['Close'], 
                       'k-', linewidth=3, label='2025 Actual', alpha=0.8)
            
            # Plot predictions for each model
            for model_name, config in self.model_configs.items():
                if model_name in predictions and predictions[model_name]['test_pred'] is not None:
                    pred_data = predictions[model_name]
                    test_dates = pred_data['test_dates']
                    test_pred = pred_data['test_pred'].flatten()
                    
                    ax.plot(test_dates, test_pred, 
                           color=config['color'], alpha=0.7, 
                           label=f'{model_name} ({config["neurons"]} neurons)', linewidth=1.5)
            
            # Formatting
            ax.set_title(f'{company_name} ({ticker})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('lstm_all_companies_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('lstm_all_companies_comparison.pdf', bbox_inches='tight')
        plt.show()
    
    def calculate_model_performance(self, all_predictions, actual_2025_data):
        """Calculate and display model performance metrics."""
        
        print("\n📊 MODEL PERFORMANCE ANALYSIS")
        print("="*80)
        
        performance_data = []
        
        for ticker in self.company_names.keys():
            if ticker not in all_predictions:
                continue
            
            company_name = self.company_names[ticker]
            predictions = all_predictions[ticker]
            
            print(f"\n📈 {company_name} ({ticker})")
            print("-" * 50)
            
            # Get actual 2025 data
            company_2025_data = actual_2025_data[actual_2025_data['Ticker'] == ticker].sort_values('Date')
            
            if len(company_2025_data) == 0:
                print("No 2025 data available for comparison")
                continue
            
            for model_name, config in self.model_configs.items():
                if model_name in predictions and predictions[model_name]['test_pred'] is not None:
                    pred_data = predictions[model_name]
                    test_pred = pred_data['test_pred'].flatten()
                    test_actual = pred_data['test_actual'].flatten()
                    
                    # Calculate metrics
                    mse = mean_squared_error(test_actual, test_pred)
                    mae = mean_absolute_error(test_actual, test_pred)
                    rmse = np.sqrt(mse)
                    
                    # Calculate percentage error
                    mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100
                    
                    # Calculate directional accuracy
                    actual_direction = np.diff(test_actual)
                    pred_direction = np.diff(test_pred)
                    directional_accuracy = np.mean(np.sign(actual_direction) == np.sign(pred_direction)) * 100
                    
                    print(f"{model_name:>10}: RMSE={rmse:6.2f}, MAE={mae:6.2f}, MAPE={mape:5.1f}%, DirAcc={directional_accuracy:5.1f}%")
                    
                    performance_data.append({
                        'Company': company_name,
                        'Ticker': ticker,
                        'Model': model_name,
                        'Neurons': config['neurons'],
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE': mape,
                        'Directional_Accuracy': directional_accuracy
                    })
        
        # Create performance summary
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            
            # Save performance data
            perf_df.to_csv('lstm_model_performance.csv', index=False)
            
            # Find best models
            print(f"\n🏆 BEST PERFORMING MODELS")
            print("-" * 50)
            
            for ticker in self.company_names.keys():
                ticker_perf = perf_df[perf_df['Ticker'] == ticker]
                if len(ticker_perf) > 0:
                    best_model = ticker_perf.loc[ticker_perf['RMSE'].idxmin()]
                    print(f"{ticker}: {best_model['Model']} ({best_model['Neurons']} neurons) - RMSE: {best_model['RMSE']:.2f}")
        
        return performance_data
    
    def run_complete_analysis(self):
        """Run the complete LSTM analysis."""
        
        print("🚀 Starting LSTM Stock Price Prediction Analysis")
        print("="*80)
        
        # Load and prepare data
        train_data, test_data = self.load_and_prepare_data()
        
        # Train models for each company
        all_trained_models = {}
        all_predictions = {}
        
        for ticker in self.company_names.keys():
            try:
                trained_models, predictions = self.train_models_for_company(ticker, train_data, test_data)
                if trained_models is not None:
                    all_trained_models[ticker] = trained_models
                    all_predictions[ticker] = predictions
                    
                    # Create individual company plots
                    self.create_prediction_plots(ticker, predictions, test_data)
                    
            except Exception as e:
                print(f"❌ Error training models for {ticker}: {str(e)}")
                continue
        
        # Create comparison plot
        if all_predictions:
            self.create_comparison_plot(all_predictions, test_data)
            
            # Calculate performance metrics
            performance_data = self.calculate_model_performance(all_predictions, test_data)
            
            print(f"\n🎉 LSTM Analysis Complete!")
            print(f"📁 Files created:")
            print(f"  📊 Individual company prediction plots (lstm_predictions_[TICKER].png/pdf)")
            print(f"  📈 All companies comparison plot (lstm_all_companies_comparison.png/pdf)")
            print(f"  📋 Model performance metrics (lstm_model_performance.csv)")
            
            return all_trained_models, all_predictions, performance_data
        
        else:
            print("❌ No models were successfully trained")
            return None, None, None

def main():
    """Main function to run LSTM analysis."""
    try:
        # Initialize predictor
        predictor = LSTMStockPredictor()
        
        # Run complete analysis
        trained_models, predictions, performance = predictor.run_complete_analysis()
        
        if trained_models is not None:
            print(f"\n✅ LSTM analysis completed successfully!")
            print(f"📊 Trained models for {len(trained_models)} companies")
            print(f"🤖 Each company has {len(predictor.model_configs)} different model configurations")
        else:
            print(f"\n❌ LSTM analysis failed")
            
    except Exception as e:
        print(f"❌ Error in LSTM analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()

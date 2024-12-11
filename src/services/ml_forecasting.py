from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import xgboost as xgb
from prophet import Prophet

logger = logging.getLogger(__name__)

class MLForecasting:
    """Advanced ML-based demand forecasting service."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.prophet_models = {}
    
    def prepare_features(
        self,
        data: pd.DataFrame,
        target_col: str = 'quantity'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for ML models.
        
        Args:
            data: DataFrame with historical data
            target_col: Target column name
            
        Returns:
            Features DataFrame and target Series
        """
        # Create time-based features
        data['year'] = data.index.year
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['day_of_week'] = data.index.dayofweek
        data['quarter'] = data.index.quarter
        
        # Create lag features
        for lag in [1, 7, 14, 30]:
            data[f'lag_{lag}'] = data[target_col].shift(lag)
        
        # Create rolling mean features
        for window in [7, 14, 30]:
            data[f'rolling_mean_{window}'] = data[target_col].rolling(window=window).mean()
            data[f'rolling_std_{window}'] = data[target_col].rolling(window=window).std()
        
        # Create seasonal features
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        data['is_month_start'] = data.index.is_month_start.astype(int)
        data['is_month_end'] = data.index.is_month_end.astype(int)
        
        # Drop NaN values created by lag features
        data = data.dropna()
        
        # Separate features and target
        features = data.drop(columns=[target_col])
        target = data[target_col]
        
        return features, target
    
    def train_models(
        self,
        item_id: int,
        historical_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Train multiple ML models for forecasting.
        
        Args:
            item_id: Item ID
            historical_data: Historical demand data
            
        Returns:
            Dictionary with model performance metrics
        """
        # Prepare data
        features, target = self.prepare_features(historical_data)
        
        # Initialize models
        models = {
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=7,
                random_state=42
            )
        }
        
        # Initialize Prophet model
        prophet_data = pd.DataFrame({
            'ds': historical_data.index,
            'y': historical_data['quantity']
        })
        prophet = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # Train-test split using TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        metrics = {
            'rf': {'mae': [], 'rmse': []},
            'xgb': {'mae': [], 'rmse': []},
            'prophet': {'mae': [], 'rmse': []}
        }
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        scaled_features = pd.DataFrame(
            scaled_features,
            columns=features.columns,
            index=features.index
        )
        
        # Train and evaluate models
        for train_idx, test_idx in tscv.split(scaled_features):
            # Split data
            X_train = scaled_features.iloc[train_idx]
            X_test = scaled_features.iloc[test_idx]
            y_train = target.iloc[train_idx]
            y_test = target.iloc[test_idx]
            
            # Train and evaluate RF and XGB
            for name, model in models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                metrics[name]['mae'].append(mean_absolute_error(y_test, predictions))
                metrics[name]['rmse'].append(np.sqrt(mean_squared_error(y_test, predictions)))
            
            # Train and evaluate Prophet
            prophet_train = prophet_data.iloc[train_idx]
            prophet_test = prophet_data.iloc[test_idx]
            
            prophet.fit(prophet_train)
            prophet_predictions = prophet.predict(pd.DataFrame({'ds': prophet_test['ds']}))
            
            metrics['prophet']['mae'].append(
                mean_absolute_error(prophet_test['y'], prophet_predictions['yhat'])
            )
            metrics['prophet']['rmse'].append(
                np.sqrt(mean_squared_error(prophet_test['y'], prophet_predictions['yhat']))
            )
        
        # Save best performing models and scaler
        self.models[item_id] = models
        self.scalers[item_id] = scaler
        self.prophet_models[item_id] = prophet
        
        if self.model_path:
            joblib.dump(models, f"{self.model_path}/models_{item_id}.joblib")
            joblib.dump(scaler, f"{self.model_path}/scaler_{item_id}.joblib")
            prophet.save(f"{self.model_path}/prophet_{item_id}.json")
        
        # Return average metrics
        return {
            model_name: {
                metric: np.mean(values)
                for metric, values in model_metrics.items()
            }
            for model_name, model_metrics in metrics.items()
        }
    
    def forecast_demand(
        self,
        item_id: int,
        features: pd.DataFrame,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate demand forecast using trained models.
        
        Args:
            item_id: Item ID
            features: Current feature values
            days: Number of days to forecast
            
        Returns:
            Dictionary with forecasts from different models
        """
        if item_id not in self.models:
            raise ValueError(f"No trained models found for item {item_id}")
        
        forecasts = {}
        
        # Scale features
        scaled_features = self.scalers[item_id].transform(features)
        
        # Generate forecasts from RF and XGB models
        for name, model in self.models[item_id].items():
            predictions = model.predict(scaled_features)
            forecasts[name] = predictions.tolist()
        
        # Generate forecast from Prophet
        future_dates = pd.DataFrame({
            'ds': pd.date_range(
                start=features.index[-1] + timedelta(days=1),
                periods=days
            )
        })
        prophet_forecast = self.prophet_models[item_id].predict(future_dates)
        forecasts['prophet'] = prophet_forecast['yhat'].tolist()
        
        # Calculate ensemble forecast (weighted average)
        weights = {
            'rf': 0.3,
            'xgb': 0.3,
            'prophet': 0.4
        }
        
        ensemble_forecast = np.zeros(days)
        for model_name, weight in weights.items():
            ensemble_forecast += np.array(forecasts[model_name]) * weight
        
        forecasts['ensemble'] = ensemble_forecast.tolist()
        
        return {
            'forecasts': forecasts,
            'confidence_intervals': {
                'lower': prophet_forecast['yhat_lower'].tolist(),
                'upper': prophet_forecast['yhat_upper'].tolist()
            }
        } 
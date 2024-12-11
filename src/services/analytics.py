from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from fastapi import HTTPException

from models.inventory import Inventory, InventoryTransaction, Warehouse
from schemas.analytics import StockAnalysis, DemandForecast

logger = logging.getLogger(__name__)

class SupplyChainAnalytics:
    """Analytics service for supply chain optimization."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def analyze_stock_levels(self, warehouse_id: Optional[int] = None) -> List[StockAnalysis]:
        """
        Analyze stock levels and identify potential stockouts or overstock situations.
        
        Args:
            warehouse_id: Optional warehouse filter
            
        Returns:
            List of stock analysis results
        """
        # Get inventory data
        query = self.db.query(Inventory)
        if warehouse_id:
            query = query.filter(Inventory.warehouse_id == warehouse_id)
        
        inventory_df = pd.DataFrame([{
            'id': item.id,
            'sku': item.sku,
            'name': item.name,
            'quantity': item.quantity,
            'reorder_point': item.reorder_point,
            'reorder_quantity': item.reorder_quantity
        } for item in query.all()])
        
        if inventory_df.empty:
            return []
        
        # Calculate stock status
        inventory_df['stock_status'] = np.where(
            inventory_df['quantity'] <= inventory_df['reorder_point'],
            'low_stock',
            np.where(
                inventory_df['quantity'] >= inventory_df['reorder_quantity'] * 2,
                'overstock',
                'optimal'
            )
        )
        
        # Calculate days until stockout based on average daily demand
        transactions_df = self._get_transaction_history()
        if not transactions_df.empty:
            avg_daily_demand = self._calculate_daily_demand(transactions_df)
            inventory_df = inventory_df.merge(
                avg_daily_demand,
                on='id',
                how='left'
            )
            inventory_df['days_until_stockout'] = np.where(
                inventory_df['avg_daily_demand'] > 0,
                inventory_df['quantity'] / inventory_df['avg_daily_demand'],
                float('inf')
            )
        else:
            inventory_df['days_until_stockout'] = float('inf')
            inventory_df['avg_daily_demand'] = 0
        
        return [
            StockAnalysis(
                item_id=row['id'],
                sku=row['sku'],
                name=row['name'],
                current_quantity=row['quantity'],
                stock_status=row['stock_status'],
                days_until_stockout=row['days_until_stockout'],
                avg_daily_demand=row['avg_daily_demand']
            )
            for _, row in inventory_df.iterrows()
        ]
    
    def forecast_demand(
        self,
        item_id: int,
        days: int = 30
    ) -> DemandForecast:
        """
        Forecast demand for an item using time series analysis.
        
        Args:
            item_id: Item ID
            days: Number of days to forecast
            
        Returns:
            Demand forecast
        """
        # Get historical transactions
        transactions = (
            self.db.query(InventoryTransaction)
            .filter(
                InventoryTransaction.item_id == item_id,
                InventoryTransaction.transaction_type == 'issue'
            )
            .order_by(InventoryTransaction.timestamp.desc())
            .all()
        )
        
        if not transactions:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for forecast"
            )
        
        # Create time series
        df = pd.DataFrame([{
            'date': tx.timestamp.date(),
            'quantity': tx.quantity
        } for tx in transactions])
        
        # Aggregate by date
        daily_demand = df.groupby('date')['quantity'].sum().reset_index()
        daily_demand = daily_demand.set_index('date')
        
        # Fill missing dates with zeros
        date_range = pd.date_range(
            start=daily_demand.index.min(),
            end=daily_demand.index.max()
        )
        daily_demand = daily_demand.reindex(date_range, fill_value=0)
        
        # Calculate moving averages
        ma7 = daily_demand['quantity'].rolling(window=7).mean()
        ma30 = daily_demand['quantity'].rolling(window=30).mean()
        
        # Simple forecast using exponential smoothing
        alpha = 0.3  # smoothing factor
        forecast = pd.Series(index=pd.date_range(
            start=daily_demand.index.max() + timedelta(days=1),
            periods=days
        ))
        
        last_value = daily_demand['quantity'].ewm(alpha=alpha).mean()[-1]
        forecast[:] = last_value
        
        # Calculate confidence intervals
        std_dev = daily_demand['quantity'].std()
        confidence_interval = 1.96 * std_dev  # 95% confidence interval
        
        return DemandForecast(
            item_id=item_id,
            forecast_days=days,
            daily_forecast=last_value,
            confidence_interval=confidence_interval,
            historical_stats={
                'mean_daily_demand': daily_demand['quantity'].mean(),
                'max_daily_demand': daily_demand['quantity'].max(),
                'min_daily_demand': daily_demand['quantity'].min(),
                'std_dev_demand': std_dev
            }
        )
    
    def _get_transaction_history(self) -> pd.DataFrame:
        """Get transaction history as DataFrame."""
        transactions = (
            self.db.query(InventoryTransaction)
            .filter(InventoryTransaction.transaction_type == 'issue')
            .all()
        )
        
        if not transactions:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            'id': tx.item_id,
            'date': tx.timestamp.date(),
            'quantity': tx.quantity
        } for tx in transactions])
    
    def _calculate_daily_demand(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate average daily demand."""
        # Group by item and calculate average daily demand
        daily_demand = transactions_df.groupby(['id', 'date'])['quantity'].sum().reset_index()
        avg_demand = daily_demand.groupby('id')['quantity'].mean().reset_index()
        avg_demand.columns = ['id', 'avg_daily_demand']
        
        return avg_demand
    
    def optimize_inventory_levels(self, warehouse_id: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        Optimize inventory levels using statistical analysis.
        
        Args:
            warehouse_id: Optional warehouse filter
            
        Returns:
            Optimization recommendations
        """
        stock_analysis = self.analyze_stock_levels(warehouse_id)
        
        recommendations = {
            'reorder_recommendations': [],
            'reduction_recommendations': []
        }
        
        for item in stock_analysis:
            if item.stock_status == 'low_stock':
                # Calculate optimal order quantity
                if item.avg_daily_demand > 0:
                    days_to_cover = 30  # Cover 30 days of demand
                    optimal_quantity = max(
                        item.avg_daily_demand * days_to_cover,
                        item.reorder_quantity
                    )
                    
                    recommendations['reorder_recommendations'].append({
                        'item_id': item.item_id,
                        'sku': item.sku,
                        'name': item.name,
                        'current_quantity': item.current_quantity,
                        'recommended_order': optimal_quantity,
                        'days_until_stockout': item.days_until_stockout
                    })
            
            elif item.stock_status == 'overstock':
                # Calculate excess inventory
                if item.avg_daily_demand > 0:
                    excess_days = item.current_quantity / item.avg_daily_demand - 60  # More than 60 days coverage
                    if excess_days > 0:
                        recommendations['reduction_recommendations'].append({
                            'item_id': item.item_id,
                            'sku': item.sku,
                            'name': item.name,
                            'current_quantity': item.current_quantity,
                            'excess_days_coverage': excess_days,
                            'potential_reduction': item.current_quantity - (item.avg_daily_demand * 60)
                        })
        
        return recommendations 
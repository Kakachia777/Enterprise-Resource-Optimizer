from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from core.deps import get_db
from schemas.analytics import (
    StockAnalysis,
    DemandForecast,
    OptimizationRecommendations
)
from services.analytics import SupplyChainAnalytics

router = APIRouter()

@router.get("/stock-analysis/", response_model=List[StockAnalysis])
def analyze_stock_levels(
    *,
    db: Session = Depends(get_db),
    warehouse_id: Optional[int] = Query(None, description="Filter by warehouse ID")
) -> List[StockAnalysis]:
    """
    Analyze current stock levels and identify potential issues.
    
    This endpoint provides detailed analysis of stock levels including:
    - Current stock status (low, optimal, overstock)
    - Days until stockout based on historical demand
    - Average daily demand
    """
    analytics = SupplyChainAnalytics(db)
    return analytics.analyze_stock_levels(warehouse_id)

@router.get("/demand-forecast/{item_id}", response_model=DemandForecast)
def forecast_item_demand(
    *,
    db: Session = Depends(get_db),
    item_id: int,
    days: int = Query(30, ge=1, le=365, description="Number of days to forecast")
) -> DemandForecast:
    """
    Generate demand forecast for specific item.
    
    Uses time series analysis to forecast future demand based on historical data.
    Provides:
    - Daily demand forecast
    - Confidence intervals
    - Historical statistics
    """
    analytics = SupplyChainAnalytics(db)
    return analytics.forecast_demand(item_id, days)

@router.get("/optimization/", response_model=OptimizationRecommendations)
def get_optimization_recommendations(
    *,
    db: Session = Depends(get_db),
    warehouse_id: Optional[int] = Query(None, description="Filter by warehouse ID")
) -> OptimizationRecommendations:
    """
    Get inventory optimization recommendations.
    
    Provides actionable recommendations for:
    - Reordering low stock items
    - Reducing excess inventory
    - Optimizing stock levels
    """
    analytics = SupplyChainAnalytics(db)
    return analytics.optimize_inventory_levels(warehouse_id)

@router.get("/dashboard/")
def get_analytics_dashboard(
    *,
    db: Session = Depends(get_db),
    warehouse_id: Optional[int] = Query(None, description="Filter by warehouse ID")
) -> dict:
    """
    Get comprehensive analytics dashboard data.
    
    Provides overview of:
    - Inventory health metrics
    - Stock level distribution
    - Value analysis
    - Demand patterns
    """
    analytics = SupplyChainAnalytics(db)
    
    # Get various analytics
    stock_analysis = analytics.analyze_stock_levels(warehouse_id)
    
    # Calculate summary metrics
    total_items = len(stock_analysis)
    low_stock_items = len([item for item in stock_analysis if item.stock_status == 'low_stock'])
    overstock_items = len([item for item in stock_analysis if item.stock_status == 'overstock'])
    
    # Calculate stock status distribution
    stock_distribution = {
        'low_stock': (low_stock_items / total_items * 100) if total_items > 0 else 0,
        'optimal': ((total_items - low_stock_items - overstock_items) / total_items * 100) if total_items > 0 else 0,
        'overstock': (overstock_items / total_items * 100) if total_items > 0 else 0
    }
    
    # Get optimization recommendations
    recommendations = analytics.optimize_inventory_levels(warehouse_id)
    
    return {
        'summary_metrics': {
            'total_items': total_items,
            'low_stock_items': low_stock_items,
            'overstock_items': overstock_items,
            'stock_health_score': (
                (total_items - low_stock_items - overstock_items) / total_items * 100
            ) if total_items > 0 else 0
        },
        'stock_distribution': stock_distribution,
        'action_required': {
            'reorder_needed': len(recommendations['reorder_recommendations']),
            'reduction_opportunities': len(recommendations['reduction_recommendations'])
        },
        'recommendations': recommendations
    } 
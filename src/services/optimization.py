from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from pulp import *
from deap import base, creator, tools, algorithms
import networkx as nx

from models.inventory import Inventory, Warehouse
from services.ml_forecasting import MLForecasting

logger = logging.getLogger(__name__)

class SupplyChainOptimizer:
    """Advanced supply chain optimization service."""
    
    def __init__(self, db_session):
        self.db = db_session
        self.forecasting = MLForecasting()
    
    def optimize_inventory_allocation(
        self,
        warehouse_ids: List[int],
        item_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Optimize inventory allocation across warehouses using linear programming.
        
        Args:
            warehouse_ids: List of warehouse IDs
            item_ids: List of item IDs
            
        Returns:
            Optimal allocation plan
        """
        # Get warehouse and item data
        warehouses = self.db.query(Warehouse).filter(Warehouse.id.in_(warehouse_ids)).all()
        items = self.db.query(Inventory).filter(Inventory.id.in_(item_ids)).all()
        
        # Create optimization problem
        prob = LpProblem("InventoryAllocation", LpMinimize)
        
        # Decision variables
        allocation = LpVariable.dicts(
            "allocation",
            ((w.id, i.id) for w in warehouses for i in items),
            lowBound=0,
            cat='Integer'
        )
        
        # Objective function: Minimize total transportation and storage costs
        transportation_costs = {
            (w.id, i.id): self._calculate_transportation_cost(w, i)
            for w in warehouses for i in items
        }
        
        storage_costs = {
            (w.id, i.id): self._calculate_storage_cost(w, i)
            for w in warehouses for i in items
        }
        
        prob += lpSum([
            allocation[w.id, i.id] * (
                transportation_costs[w.id, i.id] +
                storage_costs[w.id, i.id]
            )
            for w in warehouses for i in items
        ])
        
        # Constraints
        # Warehouse capacity
        for w in warehouses:
            prob += lpSum([
                allocation[w.id, i.id] * i.unit_size
                for i in items
            ]) <= w.total_capacity
        
        # Demand satisfaction
        for i in items:
            prob += lpSum([
                allocation[w.id, i.id]
                for w in warehouses
            ]) >= i.reorder_point
        
        # Solve problem
        prob.solve()
        
        # Extract solution
        solution = {
            'status': LpStatus[prob.status],
            'total_cost': value(prob.objective),
            'allocation': {
                f"warehouse_{w.id}_item_{i.id}": value(allocation[w.id, i.id])
                for w in warehouses for i in items
            }
        }
        
        return solution
    
    def optimize_reorder_points(
        self,
        item_ids: List[int],
        service_level: float = 0.95
    ) -> Dict[int, Dict[str, float]]:
        """
        Optimize reorder points using genetic algorithms.
        
        Args:
            item_ids: List of item IDs
            service_level: Desired service level
            
        Returns:
            Optimized reorder points
        """
        # Create genetic algorithm components
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        # Attribute generator
        toolbox.register(
            "attr_float",
            random.uniform,
            0,
            1000  # Max reorder point
        )
        
        # Structure initializers
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_float,
            n=len(item_ids)
        )
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual
        )
        
        # Genetic operators
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=100, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Fitness function
        def evaluate(individual):
            total_cost = 0
            for idx, reorder_point in enumerate(individual):
                item = self.db.query(Inventory).get(item_ids[idx])
                holding_cost = self._calculate_holding_cost(item, reorder_point)
                stockout_cost = self._calculate_stockout_cost(
                    item,
                    reorder_point,
                    service_level
                )
                total_cost += holding_cost + stockout_cost
            return total_cost,
        
        toolbox.register("evaluate", evaluate)
        
        # Run genetic algorithm
        population = toolbox.population(n=50)
        ngen = 100
        
        algorithms.eaSimple(
            population,
            toolbox,
            cxpb=0.7,  # Crossover probability
            mutpb=0.2,  # Mutation probability
            ngen=ngen,
            verbose=False
        )
        
        # Get best solution
        best_individual = tools.selBest(population, k=1)[0]
        
        return {
            item_id: {
                'reorder_point': reorder_point,
                'service_level': service_level,
                'holding_cost': self._calculate_holding_cost(
                    self.db.query(Inventory).get(item_id),
                    reorder_point
                )
            }
            for item_id, reorder_point in zip(item_ids, best_individual)
        }
    
    def optimize_warehouse_network(
        self,
        warehouses: List[Warehouse],
        demand_points: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize warehouse network using graph theory.
        
        Args:
            warehouses: List of warehouses
            demand_points: List of demand points with coordinates
            
        Returns:
            Optimized network configuration
        """
        # Create graph
        G = nx.Graph()
        
        # Add warehouse nodes
        for w in warehouses:
            G.add_node(
                f"W{w.id}",
                type='warehouse',
                capacity=w.total_capacity,
                fixed_cost=w.fixed_cost
            )
        
        # Add demand point nodes
        for idx, dp in enumerate(demand_points):
            G.add_node(
                f"D{idx}",
                type='demand',
                demand=dp['demand'],
                coords=(dp['lat'], dp['lon'])
            )
        
        # Add edges with transportation costs
        for w in warehouses:
            for idx, dp in enumerate(demand_points):
                distance = self._calculate_distance(
                    (w.latitude, w.longitude),
                    (dp['lat'], dp['lon'])
                )
                cost = distance * dp['demand'] * 0.1  # Cost per unit-km
                G.add_edge(f"W{w.id}", f"D{idx}", weight=cost)
        
        # Find minimum spanning tree
        mst = nx.minimum_spanning_tree(G)
        
        # Calculate optimal flows
        flows = nx.max_flow_min_cost(mst, 'source', 'sink')
        
        # Calculate total cost
        total_cost = sum(
            d['weight'] * flows[u][v]
            for u, v, d in mst.edges(data=True)
        )
        
        return {
            'network': {
                'nodes': list(mst.nodes(data=True)),
                'edges': list(mst.edges(data=True))
            },
            'flows': flows,
            'total_cost': total_cost
        }
    
    def _calculate_transportation_cost(
        self,
        warehouse: Warehouse,
        item: Inventory
    ) -> float:
        """Calculate transportation cost between warehouse and item."""
        # Simplified cost calculation
        distance = 100  # Example distance in km
        cost_per_unit_km = 0.1
        return distance * cost_per_unit_km * item.unit_price
    
    def _calculate_storage_cost(
        self,
        warehouse: Warehouse,
        item: Inventory
    ) -> float:
        """Calculate storage cost for item in warehouse."""
        # Simplified storage cost calculation
        storage_rate = 0.02  # 2% of item value per period
        return item.unit_price * storage_rate
    
    def _calculate_holding_cost(
        self,
        item: Inventory,
        reorder_point: float
    ) -> float:
        """Calculate inventory holding cost."""
        holding_rate = 0.2  # 20% annual holding cost rate
        average_inventory = (reorder_point + item.reorder_quantity) / 2
        return item.unit_price * holding_rate * average_inventory
    
    def _calculate_stockout_cost(
        self,
        item: Inventory,
        reorder_point: float,
        service_level: float
    ) -> float:
        """Calculate expected stockout cost."""
        # Get demand forecast
        forecast = self.forecasting.forecast_demand(item.id, pd.DataFrame())
        
        # Calculate probability of stockout
        demand_std = np.std(forecast['forecasts']['ensemble'])
        z_score = norm.ppf(service_level)
        safety_stock = z_score * demand_std
        
        stockout_prob = 1 - service_level
        stockout_cost = item.unit_price * 2  # Penalty cost per unit
        
        return stockout_prob * stockout_cost * max(0, safety_stock - reorder_point)
    
    def _calculate_distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """Calculate distance between two points using Haversine formula."""
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        R = 6371  # Earth's radius in kilometers
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (
            np.sin(dlat/2) ** 2 +
            np.cos(np.radians(lat1)) *
            np.cos(np.radians(lat2)) *
            np.sin(dlon/2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
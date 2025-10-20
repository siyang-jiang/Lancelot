"""
Aggregation methods for federated learning.

This module implements various aggregation strategies for combining
client model updates in federated learning, including Byzantine-robust methods.
"""

import copy
import random
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class BaseAggregator(ABC):
    """Base class for federated learning aggregators."""
    
    @abstractmethod
    def aggregate(self, local_models: List[Dict[str, torch.Tensor]], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Aggregate local model updates into a global model.
        
        Args:
            local_models: List of local model state dictionaries
            **kwargs: Additional aggregation-specific parameters
            
        Returns:
            Dict containing the aggregated global model state dictionary
        """
        pass


class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg) aggregator.
    
    Computes the simple average of all client model parameters.
    Not Byzantine-robust but serves as a baseline.
    """
    
    def aggregate(self, local_models: List[Dict[str, torch.Tensor]], 
                  weights: List[float] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Perform federated averaging of local models.
        
        Args:
            local_models: List of local model state dictionaries
            weights: Optional weights for weighted averaging (defaults to uniform)
            
        Returns:
            Averaged global model state dictionary
        """
        if not local_models:
            raise ValueError("No local models provided for aggregation")
        
        if weights is None:
            weights = [1.0 / len(local_models)] * len(local_models)
        
        if len(weights) != len(local_models):
            raise ValueError("Number of weights must match number of models")
        
        # Initialize with first model
        global_model = copy.deepcopy(local_models[0])
        
        with torch.no_grad():
            # Zero out the global model
            for key in global_model.keys():
                global_model[key] = torch.zeros_like(global_model[key])
            
            # Weighted sum of all models
            for i, local_model in enumerate(local_models):
                for key in global_model.keys():
                    global_model[key] += weights[i] * local_model[key]
        
        return global_model


class NoiseInjectedFedAvgAggregator(FedAvgAggregator):
    """
    FedAvg with small noise injection for differentiation.
    
    Used in some experimental settings to break ties or add slight randomness.
    """
    
    def __init__(self, noise_scale: float = 1e-4):
        """
        Initialize with noise scale.
        
        Args:
            noise_scale: Scale of uniform noise to add
        """
        self.noise_scale = noise_scale
    
    def aggregate(self, local_models: List[Dict[str, torch.Tensor]], 
                  weights: List[float] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Perform federated averaging with noise injection."""
        global_model = super().aggregate(local_models, weights, **kwargs)
        
        # Add small random noise to each parameter
        with torch.no_grad():
            for key in global_model.keys():
                noise = torch.empty_like(global_model[key]).uniform_(0, self.noise_scale)
                global_model[key] += noise
        
        return global_model


class MedianAggregator(BaseAggregator):
    """
    Coordinate-wise median aggregator.
    
    Computes the median of each parameter independently across clients.
    Provides some Byzantine robustness but may not preserve model coherence.
    """
    
    def aggregate(self, local_models: List[Dict[str, torch.Tensor]], **kwargs) -> Dict[str, torch.Tensor]:
        """Perform coordinate-wise median aggregation."""
        if not local_models:
            raise ValueError("No local models provided for aggregation")
        
        global_model = copy.deepcopy(local_models[0])
        
        with torch.no_grad():
            for key in global_model.keys():
                # Stack tensors from all models
                stacked = torch.stack([model[key] for model in local_models], dim=0)
                # Compute median along the client dimension
                global_model[key] = torch.median(stacked, dim=0)[0]
        
        return global_model


class TrimmedMeanAggregator(BaseAggregator):
    """
    Trimmed mean aggregator for Byzantine robustness.
    
    Removes outliers and averages the remaining models.
    """
    
    def __init__(self, trim_ratio: float = 0.1):
        """
        Initialize trimmed mean aggregator.
        
        Args:
            trim_ratio: Fraction of models to trim from each end
        """
        self.trim_ratio = trim_ratio
    
    def aggregate(self, local_models: List[Dict[str, torch.Tensor]], **kwargs) -> Dict[str, torch.Tensor]:
        """Perform trimmed mean aggregation."""
        if not local_models:
            raise ValueError("No local models provided for aggregation")
        
        n_models = len(local_models)
        n_trim = int(n_models * self.trim_ratio)
        
        if n_trim * 2 >= n_models:
            raise ValueError("Trim ratio too large - would remove all models")
        
        global_model = copy.deepcopy(local_models[0])
        
        with torch.no_grad():
            for key in global_model.keys():
                # Stack tensors from all models
                stacked = torch.stack([model[key] for model in local_models], dim=0)
                
                # Sort along client dimension
                sorted_tensors, _ = torch.sort(stacked, dim=0)
                
                # Take the middle portion (trim from both ends)
                trimmed = sorted_tensors[n_trim:n_models-n_trim]
                
                # Average the remaining tensors
                global_model[key] = torch.mean(trimmed, dim=0)
        
        return global_model


# Legacy functions for backward compatibility
def fedavg(w_locals: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Legacy FedAvg function for backward compatibility.
    
    Args:
        w_locals: List of local model state dictionaries
        
    Returns:
        Averaged global model state dictionary
    """
    aggregator = FedAvgAggregator()
    return aggregator.aggregate(w_locals)


def fedavg_5wei(w_locals: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Legacy FedAvg with noise function for backward compatibility.
    
    Args:
        w_locals: List of local model state dictionaries
        
    Returns:
        Averaged global model state dictionary with small noise
    """
    aggregator = NoiseInjectedFedAvgAggregator(noise_scale=1e-4)
    return aggregator.aggregate(w_locals)


def create_aggregator(method: str, **kwargs) -> BaseAggregator:
    """
    Factory function to create aggregators by name.
    
    Args:
        method: Name of the aggregation method
        **kwargs: Additional parameters for the aggregator
        
    Returns:
        Initialized aggregator instance
    """
    aggregators = {
        'fedavg': FedAvgAggregator,
        'fedavg_noise': NoiseInjectedFedAvgAggregator,
        'median': MedianAggregator,
        'trimmed_mean': TrimmedMeanAggregator,
    }
    
    if method not in aggregators:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return aggregators[method](**kwargs)


def get_available_aggregators() -> List[str]:
    """Get list of available aggregation methods."""
    return ['fedavg', 'fedavg_noise', 'median', 'trimmed_mean']
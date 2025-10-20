"""
Local client update modules for federated learning.

This module implements different types of client behaviors including
benign clients and various Byzantine attack strategies.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from utils.dataset import DatasetSplit


class BaseLocalUpdate(ABC):
    """
    Base class for local client updates in federated learning.
    
    Defines the interface that all client update strategies must implement.
    """
    
    def __init__(self, args, dataset=None, idxs=None):
        """
        Initialize the local update handler.
        
        Args:
            args: Configuration arguments
            dataset: Local dataset for training
            idxs: Indices of data samples assigned to this client
        """
        self.args = args
        self.device = args.device
        self.loss_func = self._get_loss_function()
        
        if dataset is not None and idxs is not None:
            self.dataloader = self._create_dataloader(dataset, idxs)
        else:
            self.dataloader = None
    
    def _get_loss_function(self) -> nn.Module:
        """Get the appropriate loss function based on dataset."""
        # Can be extended for different loss functions per dataset
        return nn.CrossEntropyLoss()
    
    def _create_dataloader(self, dataset, idxs) -> DataLoader:
        """Create a DataLoader for the local dataset."""
        local_dataset = DatasetSplit(dataset, idxs)
        return DataLoader(
            local_dataset, 
            batch_size=self.args.local_bs, 
            shuffle=True, 
            drop_last=True
        )
    
    def _get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Get the optimizer for local training.
        
        Args:
            model: The neural network model
            
        Returns:
            Configured optimizer
        """
        # Using Adam optimizer as default - can be made configurable
        return torch.optim.Adam(
            model.parameters(), 
            lr=self.args.lr,
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=0
        )
    
    def _process_batch(self, images: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Process a batch of data according to dataset requirements.
        
        Args:
            images: Input images
            labels: Target labels
            
        Returns:
            Tuple of (processed_images, processed_labels)
        """
        images = images.to(self.device)
        
        # Handle different label formats for different datasets
        if self.args.dataset in ['chestmnist']:
            labels = labels.to(self.device, dtype=torch.float32)
        else:
            labels = labels.to(self.device)
            if labels.dim() > 1:
                labels = labels.squeeze(dim=-1)
        
        return images, labels
    
    @abstractmethod
    def train(self, net: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Perform local training and return updated model parameters.
        
        Args:
            net: Neural network model to train
            
        Returns:
            Updated model state dictionary
        """
        pass


class BenignUpdate(BaseLocalUpdate):
    """
    Benign (honest) client update implementation.
    
    Performs standard local training without any malicious behavior.
    """
    
    def train(self, net: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Perform benign local training.
        
        Args:
            net: Neural network model to train locally
            
        Returns:
            Updated model state dictionary after local training
        """
        if self.dataloader is None:
            raise ValueError("No dataloader available for training")
        
        net.train()
        optimizer = self._get_optimizer(net)
        
        # Local training for specified number of epochs
        for epoch in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.dataloader):
                # Process batch according to dataset requirements
                images, labels = self._process_batch(images, labels)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = net(images)
                loss = self.loss_func(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
        
        return net.state_dict()


class CompromisedUpdate(BaseLocalUpdate):
    """
    Compromised (Byzantine) client implementing model poisoning attacks.
    
    Performs local training but then applies a targeted poisoning attack
    by scaling the difference between original and trained parameters.
    """
    
    def __init__(self, args, dataset=None, idxs=None):
        """
        Initialize compromised client.
        
        Args:
            args: Configuration arguments (must include mp_alpha for attack strength)
            dataset: Local dataset
            idxs: Data indices
        """
        super().__init__(args, dataset, idxs)
        self.attack_strength = getattr(args, 'mp_alpha', 10.0)
    
    def train(self, net: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Perform compromised local training with model poisoning.
        
        Args:
            net: Neural network model to train and poison
            
        Returns:
            Poisoned model state dictionary
        """
        if self.dataloader is None:
            raise ValueError("No dataloader available for training")
        
        # Keep a copy of the original model
        net_original = copy.deepcopy(net)
        original_params = {name: param.clone() for name, param in net_original.named_parameters()}
        
        net.train()
        optimizer = self._get_optimizer(net)
        
        # Perform normal local training first
        for epoch in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.dataloader):
                # Process batch
                images, labels = self._process_batch(images, labels)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = net(images)
                loss = self.loss_func(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
        
        # Apply model poisoning attack
        poisoned_state_dict = self._apply_poisoning_attack(
            original_params, 
            net.state_dict()
        )
        
        return poisoned_state_dict
    
    def _apply_poisoning_attack(self, original_params: Dict[str, torch.Tensor], 
                               trained_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply model poisoning by amplifying parameter changes.
        
        Args:
            original_params: Original model parameters before training
            trained_params: Model parameters after local training
            
        Returns:
            Poisoned model parameters
        """
        poisoned_params = {}
        
        for name, trained_param in trained_params.items():
            if name in original_params:
                original_param = original_params[name]
                # Amplify the change: poisoned = original + alpha * (trained - original)
                # This is equivalent to: poisoned = (trained - original) * alpha
                param_diff = trained_param - original_param
                poisoned_params[name] = original_param + self.attack_strength * param_diff
            else:
                # If parameter not in original (shouldn't happen), keep as is
                poisoned_params[name] = trained_param
        
        return poisoned_params


class GradientReversalUpdate(BaseLocalUpdate):
    """
    Gradient reversal attack implementation.
    
    Performs training with reversed gradients to maximize loss instead of minimizing it.
    """
    
    def train(self, net: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Perform gradient reversal attack.
        
        Args:
            net: Neural network model
            
        Returns:
            Model parameters after gradient reversal attack
        """
        if self.dataloader is None:
            raise ValueError("No dataloader available for training")
        
        net.train()
        optimizer = self._get_optimizer(net)
        
        for epoch in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.dataloader):
                images, labels = self._process_batch(images, labels)
                
                optimizer.zero_grad()
                outputs = net(images)
                loss = self.loss_func(outputs, labels)
                
                # Reverse the gradient direction
                loss = -loss
                
                loss.backward()
                optimizer.step()
        
        return net.state_dict()


class LabelFlippingUpdate(BaseLocalUpdate):
    """
    Label flipping attack implementation.
    
    Trains with flipped/corrupted labels to degrade model performance.
    """
    
    def __init__(self, args, dataset=None, idxs=None, flip_ratio: float = 0.5):
        """
        Initialize label flipping attack.
        
        Args:
            args: Configuration arguments
            dataset: Local dataset
            idxs: Data indices
            flip_ratio: Fraction of labels to flip
        """
        super().__init__(args, dataset, idxs)
        self.flip_ratio = flip_ratio
    
    def _flip_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Randomly flip a fraction of labels.
        
        Args:
            labels: Original labels
            
        Returns:
            Labels with some entries flipped
        """
        flipped_labels = labels.clone()
        batch_size = labels.size(0)
        num_flip = int(batch_size * self.flip_ratio)
        
        if num_flip > 0:
            # Randomly select indices to flip
            flip_indices = torch.randperm(batch_size)[:num_flip]
            
            # Flip to random labels (different from original)
            for idx in flip_indices:
                original_label = labels[idx].item()
                new_label = (original_label + torch.randint(1, self.args.num_classes, (1,)).item()) % self.args.num_classes
                flipped_labels[idx] = new_label
        
        return flipped_labels
    
    def train(self, net: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Perform training with label flipping attack.
        
        Args:
            net: Neural network model
            
        Returns:
            Model parameters after training with flipped labels
        """
        if self.dataloader is None:
            raise ValueError("No dataloader available for training")
        
        net.train()
        optimizer = self._get_optimizer(net)
        
        for epoch in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.dataloader):
                images, labels = self._process_batch(images, labels)
                
                # Flip some labels
                corrupted_labels = self._flip_labels(labels)
                
                optimizer.zero_grad()
                outputs = net(images)
                loss = self.loss_func(outputs, corrupted_labels)
                
                loss.backward()
                optimizer.step()
        
        return net.state_dict()


def create_local_updater(client_type: str, args, dataset=None, idxs=None, **kwargs) -> BaseLocalUpdate:
    """
    Factory function to create local update handlers.
    
    Args:
        client_type: Type of client ('benign', 'compromised', 'gradient_reversal', 'label_flipping')
        args: Configuration arguments
        dataset: Local dataset
        idxs: Data indices
        **kwargs: Additional parameters for specific updater types
        
    Returns:
        Initialized local update handler
    """
    updaters = {
        'benign': BenignUpdate,
        'compromised': CompromisedUpdate,
        'gradient_reversal': GradientReversalUpdate,
        'label_flipping': LabelFlippingUpdate,
    }
    
    if client_type not in updaters:
        raise ValueError(f"Unknown client type: {client_type}")
    
    return updaters[client_type](args, dataset, idxs, **kwargs)


def get_available_client_types() -> list:
    """Get list of available client types."""
    return ['benign', 'compromised', 'gradient_reversal', 'label_flipping']
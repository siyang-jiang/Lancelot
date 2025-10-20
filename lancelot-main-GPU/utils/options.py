"""
Configuration parser for Lancelot federated learning system.

This module defines all command-line arguments and configuration options
for running Byzantine-robust federated learning experiments.
"""

import argparse
from typing import Any


class LancelotConfig:
    """Configuration class for Lancelot with validation and defaults."""
    
    def __init__(self, args):
        self.args = args
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.args.c_frac < 0 or self.args.c_frac > 1:
            raise ValueError("Compromised fraction must be between 0 and 1")
        
        if self.args.frac <= 0 or self.args.frac > 1:
            raise ValueError("Client fraction must be between 0 and 1")
        
        if self.args.num_clients <= 0:
            raise ValueError("Number of clients must be positive")
        
        if self.args.global_ep <= 0:
            raise ValueError("Number of global epochs must be positive")


def args_parser():
    """
    Parse command line arguments for Lancelot federated learning system.
    
    Returns:
        argparse.Namespace: Parsed arguments with all configuration options
    """
    parser = argparse.ArgumentParser(
        description="Lancelot: Byzantine-Robust Federated Learning with FHE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # =============================================================================
    # Federated Learning Configuration
    # =============================================================================
    fl_group = parser.add_argument_group('Federated Learning')
    
    fl_group.add_argument(
        '--method', 
        type=str, 
        default='krum', 
        choices=['krum', 'trimmed_mean', 'fang', 'fedavg'],
        help="Aggregation method for Byzantine robustness"
    )
    
    fl_group.add_argument(
        '--global_ep', 
        type=int, 
        default=100, 
        help="Total number of communication rounds"
    )
    
    fl_group.add_argument(
        '--num_clients', 
        type=int, 
        default=4, 
        help="Total number of federated clients"
    )
    
    fl_group.add_argument(
        '--frac', 
        type=float, 
        default=1.0, 
        help="Fraction of clients participating in each round"
    )
    
    fl_group.add_argument(
        '--local_ep', 
        type=int, 
        default=5, 
        help="Number of local training epochs per client"
    )
    
    fl_group.add_argument(
        '--local_bs', 
        type=int, 
        default=20, 
        help="Local batch size for client training"
    )
    
    fl_group.add_argument(
        '--lr', 
        type=float, 
        default=0.001, 
        help="Learning rate for client optimization"
    )
    
    # =============================================================================
    # Data and Model Configuration
    # =============================================================================
    data_group = parser.add_argument_group('Data and Model')
    
    data_group.add_argument(
        '--dataset', 
        type=str, 
        default='CIFAR10',
        choices=['CIFAR10', 'MNIST', 'FaMNIST', 'SVHN', 'ImageNet'],
        help="Dataset for federated learning"
    )
    
    data_group.add_argument(
        '--model', 
        type=str, 
        default='resnet', 
        help="Model architecture (automatically selected based on dataset)"
    )
    
    data_group.add_argument(
        '--num_classes', 
        type=int, 
        default=10, 
        help="Number of classes in the dataset"
    )
    
    data_group.add_argument(
        '--sampling', 
        type=str, 
        default='noniid', 
        choices=['noniid', 'iid'],
        help="Data distribution method across clients"
    )
    
    data_group.add_argument(
        '--alpha', 
        type=float, 
        default=10.0, 
        help="Dirichlet distribution parameter for non-IID data"
    )
    
    data_group.add_argument(
        '--num_data', 
        type=int, 
        default=100, 
        help="Number of data samples per client for label skew"
    )
    
    data_group.add_argument(
        '--quantity_skew', 
        action='store_true', 
        help="Enable quantity skew in data distribution"
    )
    
    data_group.add_argument(
        '--ratio', 
        type=float, 
        default=1.0, 
        help="Ratio of dataset size to use"
    )
    
    # =============================================================================
    # Byzantine Attack Configuration
    # =============================================================================
    attack_group = parser.add_argument_group('Byzantine Attacks')
    
    attack_group.add_argument(
        '--c_frac', 
        type=float, 
        default=0.0, 
        help="Fraction of compromised (Byzantine) clients"
    )
    
    attack_group.add_argument(
        '--p', 
        type=str, 
        default='normal',
        choices=['normal', 'target', 'untarget'],
        help="Attack type: normal, targeted, or untargeted poisoning"
    )
    
    attack_group.add_argument(
        '--mp_alpha', 
        type=float, 
        default=10.0, 
        help="Hyperparameter for targeted model poisoning attack"
    )
    
    attack_group.add_argument(
        '--mp_lambda', 
        type=float, 
        default=10.0, 
        help="Hyperparameter for untargeted model poisoning attack"
    )
    
    # =============================================================================
    # Privacy and Encryption Configuration
    # =============================================================================
    privacy_group = parser.add_argument_group('Privacy and Encryption')
    
    privacy_group.add_argument(
        '--cipher_open', 
        type=int, 
        default=0,
        choices=[0, 1],
        help="Enable homomorphic encryption (Lancelot): 0=disabled, 1=enabled"
    )
    
    privacy_group.add_argument(
        '--openfhe', 
        type=bool, 
        default=False, 
        help="Use OpenFHE library for comparison benchmarks"
    )
    
    privacy_group.add_argument(
        '--checks', 
        type=bool, 
        default=False, 
        help="Enable correctness checks between plaintext and encrypted results"
    )
    
    # =============================================================================
    # Training and System Configuration
    # =============================================================================
    system_group = parser.add_argument_group('System and Training')
    
    system_group.add_argument(
        '--gpu', 
        type=int, 
        default=0, 
        help="GPU ID to use (-1 for CPU)"
    )
    
    system_group.add_argument(
        '--seed', 
        type=int, 
        default=3, 
        help="Random seed for reproducible results"
    )
    
    system_group.add_argument(
        '--bs', 
        type=int, 
        default=20, 
        help="Batch size for testing"
    )
    
    system_group.add_argument(
        '--ds', 
        type=int, 
        default=20, 
        help="Dummy batch size for some operations"
    )
    
    # =============================================================================
    # Logging and Monitoring Configuration
    # =============================================================================
    logging_group = parser.add_argument_group('Logging and Monitoring')
    
    logging_group.add_argument(
        '--tsboard', 
        action='store_true', 
        help="Enable TensorBoard logging"
    )
    
    logging_group.add_argument(
        '--debug', 
        action='store_true', 
        help="Enable debug mode with additional logging"
    )
    
    # =============================================================================
    # Early Stopping Configuration
    # =============================================================================
    early_stop_group = parser.add_argument_group('Early Stopping')
    
    early_stop_group.add_argument(
        '--earlystop', 
        action='store_true', 
        help="Enable early stopping based on validation performance"
    )
    
    early_stop_group.add_argument(
        '--patience', 
        type=int, 
        default=8, 
        help="Number of rounds to wait before early stopping"
    )
    
    early_stop_group.add_argument(
        '--delta', 
        type=float, 
        default=0.01, 
        help="Minimum improvement threshold for early stopping"
    )
    
    # =============================================================================
    # Legacy/Experimental Parameters
    # =============================================================================
    exp_group = parser.add_argument_group('Experimental')
    
    exp_group.add_argument(
        '--num_pretrain', 
        type=int, 
        default=50, 
        help="Number of data samples for pretraining (experimental)"
    )
    
    args = parser.parse_args()
    
    # Create configuration object with validation
    config = LancelotConfig(args)
    
    return args


def print_config(args):
    """
    Print the current configuration in a readable format.
    
    Args:
        args: Parsed arguments from args_parser()
    """
    print("="*60)
    print("LANCELOT CONFIGURATION")
    print("="*60)
    
    print(f"🔧 Federated Learning:")
    print(f"   Method: {args.method}")
    print(f"   Global epochs: {args.global_ep}")
    print(f"   Clients: {args.num_clients} (fraction: {args.frac})")
    print(f"   Local epochs: {args.local_ep}")
    print(f"   Learning rate: {args.lr}")
    
    print(f"\n📊 Data:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Classes: {args.num_classes}")
    print(f"   Distribution: {args.sampling} (alpha: {args.alpha})")
    
    print(f"\n🛡️ Security:")
    print(f"   Byzantine fraction: {args.c_frac}")
    print(f"   Attack type: {args.p}")
    print(f"   Encryption: {'Enabled' if args.cipher_open else 'Disabled'}")
    
    print(f"\n💻 System:")
    print(f"   GPU: {args.gpu}")
    print(f"   Seed: {args.seed}")
    print(f"   TensorBoard: {'Enabled' if args.tsboard else 'Disabled'}")
    
    print("="*60)


if __name__ == "__main__":
    # Example usage
    args = args_parser()
    print_config(args)
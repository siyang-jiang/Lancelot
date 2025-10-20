"""
Lancelot: Byzantine-Robust Federated Learning with Fully Homomorphic Encryption

Main training script for federated learning with Byzantine robustness and privacy protection.
Supports both plaintext and encrypted aggregation methods.
"""

import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, resnet34, resnet50, mobilenet_v2
import time
import copy
import numpy as np
import random
import os
from tqdm import trange
from typing import Dict, List, Tuple, Optional, Any

from utils.options import args_parser
from utils.sampling import noniid
from utils.dataset import load_data, LeNet5
from utils.test import test_img
from utils.byzantine_fl import GPU_krum, krum, trimmed_mean, fang, dummy_contrastive_aggregation
from utils.openfhe import set_parameters
from utils.attack import compromised_clients, untargeted_attack
from src.aggregation import fedavg, fedavg_5wei
from src.update import BenignUpdate, CompromisedUpdate


class LancelotTrainer:
    """
    Main trainer class for Lancelot federated learning system.
    
    Handles both plaintext and encrypted federated learning with Byzantine robustness.
    """
    
    def __init__(self, args):
        """Initialize the Lancelot trainer with configuration."""
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        
        # Set random seeds for reproducibility
        self._set_seed(args.seed)
        
        # Initialize logging
        self.writer = None
        if args.tsboard:
            self.writer = SummaryWriter(f'runs/data')
        
        # Initialize metrics tracking
        self.metrics = {
            'test_accuracies': [],
            'test_losses': [],
            'round_times': [],
            'local_train_time': 0.0
        }
        
        # Early stopping variables
        self.early_stop_counter = 0
        self.best_accuracy = 0.0
        
    def _set_seed(self, seed: int = 42) -> None:
        """Set random seeds for reproducible results."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        
    def _get_model(self) -> torch.nn.Module:
        """Get the appropriate model based on dataset."""
        models_map = {
            ("MNIST", "pathmnist", "pneumoniamnist", "tissuemnist"): 
                lambda: LeNet5(),
            ("FaMNIST", "chestmnist", "dermamnist", "retinamnist", "organamnist"): 
                lambda: resnet18(num_classes=self.args.num_classes),
            ("CIFAR10", "dermamnist", "breastmnist", "organcmnist"): 
                lambda: resnet34(num_classes=self.args.num_classes),
            ("SVHN", "octmnist", "bloodmnist", "organsmnist"): 
                lambda: resnet50(num_classes=self.args.num_classes),
            ("ImageNet",): 
                lambda: mobilenet_v2(num_classes=self.args.num_classes)
        }
        
        for datasets, model_fn in models_map.items():
            if self.args.dataset in datasets:
                return model_fn().to(self.device)
        
        raise ValueError(f"Unsupported dataset: {self.args.dataset}")
    
    def _setup_logging(self) -> str:
        """Setup logging file based on configuration."""
        if self.args.cipher_open:
            return f'log_cipher_{self.args.dataset}_{self.args.seed}.txt'
        else:
            return f'log_{self.args.dataset}_{self.args.seed}.txt'
    
    def _select_clients(self) -> Tuple[List[int], int]:
        """Select clients for current round and determine compromised count."""
        selected_clients = max(int(self.args.frac * self.args.num_clients), 1)
        compromised_num = int(self.args.c_frac * selected_clients)
        idxs_users = np.random.choice(range(self.args.num_clients), selected_clients, replace=False)
        return idxs_users, compromised_num
    
    def _collect_local_updates(self, net_glob: torch.nn.Module, dict_users: Dict, 
                              idxs_users: List[int], compromised_idxs: List[int], 
                              dataset_train) -> List[Dict]:
        """Collect local model updates from selected clients."""
        w_locals = []
        
        for idx in idxs_users:
            if idx in compromised_idxs:
                if self.args.p == "untarget":
                    # Untargeted attack
                    w_locals.append(copy.deepcopy(untargeted_attack(net_glob.state_dict(), self.args)))
                else:
                    # Targeted attack
                    local = CompromisedUpdate(args=self.args, dataset=dataset_train, idxs=dict_users[idx])
                    w = local.train(net=copy.deepcopy(net_glob).to(self.device))
                    w_locals.append(copy.deepcopy(w))
            else:
                # Benign client update
                local = BenignUpdate(args=self.args, dataset=dataset_train, idxs=dict_users[idx])
                start_time = time.time()
                w = local.train(net=copy.deepcopy(net_glob).to(self.device))
                end_time = time.time()
                self.metrics['local_train_time'] += end_time - start_time
                w_locals.append(copy.deepcopy(w))
        
        return w_locals
    
    def _aggregate_updates(self, w_locals: List[Dict], compromised_num: int, 
                          cryptocontext=None, keypair=None) -> Tuple[Dict, float]:
        """Aggregate local updates using specified method."""
        if self.args.method != 'krum':
            raise ValueError('Error: unrecognized aggregation technique')
        
        # Plaintext aggregation
        w_glob_plain, chosen_idx, round_time = krum(w_locals, compromised_num, self.args, 
                                                   cryptocontext, keypair)
        
        if self.args.openfhe:
            print(f"\n ==> One Round OpenFHE Time is {round_time:.6f} seconds.")
        else:
            print(f"\n ==> Local Training Krum Time is {round_time:.6f} seconds.")
        
        # Encrypted aggregation (Lancelot)
        if self.args.cipher_open:
            print("\n +------ Train on the ciphertext (Lancelot). -------+")
            start = time.time()
            w_glob_flat = GPU_krum(w_locals, compromised_num, self.args)
            end = time.time()
            print(f"==> One Round Lancelot Time is {end-start:.6f} seconds.")
            
            w_glob_encrypted = self._reshape_flat_to_state_dict(w_glob_flat, w_glob_plain)
            
            if self.args.checks:
                error = self._compute_model_error(w_glob_plain, w_glob_encrypted)
                print(f"Model error between plaintext and encrypted: {error:.6f}")
            
            return w_glob_encrypted, end - start
        
        return w_glob_plain, round_time
    
    def _reshape_flat_to_state_dict(self, flat_list: List, state_dict_template: Dict) -> Dict:
        """Reshape flat parameter list back to state dict format."""
        new_state_dict = {}
        index = 0
        for key, value in state_dict_template.items():
            numel = value.numel()
            new_state_dict[key] = torch.tensor(
                flat_list[index:index+numel], 
                dtype=value.dtype, 
                device=value.device
            ).reshape(value.shape)
            index += numel
        return new_state_dict
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _compute_model_error(self, model1: Dict, model2: Dict) -> float:
        """Compute error between two model state dictionaries."""
        flattened_model1 = self._flatten_dict(model1)
        flattened_model2 = self._flatten_dict(model2)
        
        model_list_1 = list(flattened_model1.values())
        model_list_2 = list(flattened_model2.values())
        
        total_error = 0.0
        total_elements = 0
        
        for x, y in zip(model_list_1, model_list_2):
            num_elements = x.numel()
            total_elements += num_elements
            
            x_cpu = x.cpu().detach()
            y_cpu = y.cpu().detach()
            error = torch.norm(x_cpu.float() - y_cpu.float()).item()
            total_error += error / num_elements
        
        return total_error / len(model_list_1)
    
    def _check_early_stopping(self, test_acc: float) -> bool:
        """Check if early stopping criteria is met."""
        if self.best_accuracy == 0:
            self.best_accuracy = test_acc
        elif test_acc < self.best_accuracy + self.args.delta:
            self.early_stop_counter += 1
        else:
            self.best_accuracy = test_acc
            self.early_stop_counter = 0
        
        return self.early_stop_counter >= self.args.patience
    
    def _log_results(self, epoch: int, test_acc: float, test_loss: float, log_file: str) -> None:
        """Log training results to file and tensorboard."""
        # File logging
        with open(log_file, "a") as f:
            log_message = f"==> EP: {epoch}, Test acc: {test_acc:.4f}"
            f.write(log_message + "\n")
        
        print(f"==> EP: {epoch}, Test acc: {test_acc:.4f}")
        
        # TensorBoard logging
        if self.writer:
            experiment_name = f'{self.args.method}_{self.args.p}_cfrac_{self.args.c_frac}_alpha_{self.args.alpha}'
            self.writer.add_scalar(f'testacc/{experiment_name}', test_acc, epoch)
            self.writer.add_scalar(f'testloss/{experiment_name}', test_loss, epoch)
        
        # Store metrics
        self.metrics['test_accuracies'].append(test_acc)
        self.metrics['test_losses'].append(test_loss)
    
    def train(self) -> Dict[str, Any]:
        """Main training loop for federated learning."""
        print("Initializing Lancelot federated learning...")
        
        # Load data and setup
        dataset_train, dataset_test, dataset_val = load_data(self.args)
        dict_users = noniid(dataset_train, self.args)
        
        # Initialize model
        net_glob = self._get_model()
        net_glob.train()
        
        # Setup compromised clients
        if self.args.c_frac > 0:
            compromised_idxs = compromised_clients(self.args)
        else:
            compromised_idxs = []
        
        # Setup encryption if needed
        cryptocontext, keypair = None, None
        if self.args.openfhe or self.args.cipher_open:
            cryptocontext, keypair = set_parameters()
        
        # Setup logging
        log_file = self._setup_logging()
        
        print(f"Starting federated training with {self.args.num_clients} clients...")
        print(f"Compromised fraction: {self.args.c_frac}")
        print(f"Method: {self.args.method}")
        print(f"Encryption enabled: {bool(self.args.cipher_open)}")
        
        # Main training loop
        for epoch in trange(self.args.global_ep, desc="Training Progress"):
            # Select clients for this round
            idxs_users, compromised_num = self._select_clients()
            
            # Collect local updates
            w_locals = self._collect_local_updates(
                net_glob, dict_users, idxs_users, compromised_idxs, dataset_train
            )
            
            # Aggregate updates
            w_glob, round_time = self._aggregate_updates(
                w_locals, compromised_num, cryptocontext, keypair
            )
            self.metrics['round_times'].append(round_time)
            
            # Update global model
            net_glob.load_state_dict(w_glob)
            
            # Evaluate model
            test_acc, test_loss = test_img(net_glob.to(self.device), dataset_test, self.args)
            
            # Log results
            self._log_results(epoch, test_acc, test_loss, log_file)
            
            # Check early stopping
            if self._check_early_stopping(test_acc):
                print('Early stopped federated training!')
                break
        
        # Clean up
        if self.writer:
            self.writer.close()
        
        print(f"Training completed!")
        print(f"Total local training time: {self.metrics['local_train_time']:.2f} seconds")
        print(f"Average round time: {np.mean(self.metrics['round_times']):.4f} seconds")
        print(f"Final test accuracy: {self.metrics['test_accuracies'][-1]:.4f}")
        
        return self.metrics


def main():
    """Main entry point."""
    # Parse arguments
    args = args_parser()
    
    # Set dataset-specific parameters
    if args.dataset in ["CIFAR10", "MNIST", "FaMNIST", "SVHN"]:
        args.num_classes = 10
        print(f"Number of classes: {args.num_classes}")
    
    # Initialize and run trainer
    trainer = LancelotTrainer(args)
    metrics = trainer.train()
    
    return metrics


if __name__ == '__main__':
    main()
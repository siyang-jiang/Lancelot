# Lancelot Developer Guide

## 🏗️ Architecture Overview

Lancelot is designed with a modular architecture that separates concerns and makes the system easy to understand and extend. Here's how the components interact:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Main Trainer  │────│  Configuration  │────│    Logging      │
│  (main.py)      │    │  (options.py)   │    │ (TensorBoard)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Local Updates   │    │  Data Loading   │
│  (update.py)    │    │  (dataset.py)   │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│  Aggregation    │    │ Byzantine FL    │
│(aggregation.py) │────│(byzantine_fl.py)│
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   Encryption    │    │    Attacks      │
│  (openfhe.py)   │    │  (attack.py)    │
└─────────────────┘    └─────────────────┘
```

## 📂 Project Structure

### Core Components

- **`main.py`**: Entry point and main training orchestrator
- **`src/`**: Core federated learning logic
  - `aggregation.py`: Model aggregation strategies
  - `update.py`: Client update implementations
- **`utils/`**: Utility modules
  - `options.py`: Configuration management
  - `dataset.py`: Data loading and preprocessing
  - `byzantine_fl.py`: Byzantine-robust algorithms
  - `attack.py`: Attack implementations
  - `openfhe.py`: Homomorphic encryption interface
  - `test.py`: Model evaluation utilities

### CAHEL Integration

- **`cahel-main/`**: GPU-accelerated homomorphic encryption library
  - `include/`: C++ header files
  - `src/`: Core CAHEL implementation
  - `python/`: Python bindings for CAHEL

## 🔧 Extending Lancelot

### Adding New Aggregation Methods

1. Create a new aggregator class in `src/aggregation.py`:

```python
class MyCustomAggregator(BaseAggregator):
    def __init__(self, custom_param=1.0):
        self.custom_param = custom_param
    
    def aggregate(self, local_models, **kwargs):
        # Your custom aggregation logic here
        return aggregated_model
```

2. Register it in the factory function:

```python
aggregators = {
    'fedavg': FedAvgAggregator,
    'my_method': MyCustomAggregator,  # Add this line
    # ... other methods
}
```

3. Update `utils/options.py` to include your method in choices:

```python
fl_group.add_argument(
    '--method', 
    choices=['krum', 'trimmed_mean', 'my_method'],  # Add here
    # ...
)
```

### Adding New Attack Types

1. Create a new attack class in `src/update.py`:

```python
class MyAttackUpdate(BaseLocalUpdate):
    def train(self, net):
        # Implement your attack strategy
        return modified_state_dict
```

2. Register in the factory function:

```python
updaters = {
    'benign': BenignUpdate,
    'my_attack': MyAttackUpdate,  # Add this
    # ...
}
```

### Adding New Datasets

1. Extend the model selection in `main.py`:

```python
models_map = {
    ("MyDataset",): lambda: MyCustomModel(num_classes=self.args.num_classes),
    # ... existing mappings
}
```

2. Update data loading in `utils/dataset.py` to handle your dataset

3. Add dataset choice in `utils/options.py`

### Adding New Defense Mechanisms

1. Implement defense in `utils/byzantine_fl.py`:

```python
def my_defense_method(w_locals, c, args):
    # Your defense implementation
    return selected_model, chosen_idx, computation_time
```

2. Integrate in the trainer's `_aggregate_updates` method

## 🧪 Running Experiments

### Basic Experiment

```bash
python main.py \
    --dataset CIFAR10 \
    --num_clients 10 \
    --c_frac 0.2 \
    --method krum \
    --global_ep 100
```

### With Encryption (Lancelot)

```bash
python main.py \
    --dataset CIFAR10 \
    --cipher_open 1 \
    --checks 1 \
    --num_clients 4 \
    --c_frac 0.25
```

### Comparing Methods

```bash
# Run multiple experiments with different methods
for method in krum trimmed_mean fedavg; do
    python main.py --method $method --dataset CIFAR10 --seed 42
done
```

### Hyperparameter Sweeps

```bash
# Test different Byzantine ratios
for c_frac in 0.0 0.1 0.2 0.3; do
    python main.py --c_frac $c_frac --dataset MNIST --global_ep 50
done
```

## 🔍 Debugging and Development

### Enable Debug Mode

```bash
python main.py --debug --checks 1
```

### Monitor with TensorBoard

```bash
python main.py --tsboard
tensorboard --logdir runs/
```

### Profile Performance

```python
# Add timing to your custom methods
import time
start_time = time.time()
# Your code here
print(f"Operation took {time.time() - start_time:.4f} seconds")
```

## 📊 Understanding Results

### Log Files

- `log_<dataset>_<seed>.txt`: Basic training logs
- `log_cipher_<dataset>_<seed>.txt`: Encrypted training logs

### TensorBoard Metrics

- `testacc/`: Test accuracy over rounds
- `testloss/`: Test loss over rounds
- Custom metrics can be added via `self.writer.add_scalar()`

### Performance Metrics

The system tracks:
- Local training time
- Aggregation time
- Encryption/decryption overhead
- Communication rounds
- Model accuracy

## 🛡️ Security Considerations

### Byzantine Robustness

- **Krum**: Selects model closest to majority
- **Trimmed Mean**: Removes outliers, averages remainder
- **Median**: Coordinate-wise median (less effective)

### Privacy Protection

- **CAHEL**: GPU-accelerated homomorphic encryption
- **OpenFHE**: CPU-based comparison baseline
- Supports CKKS scheme for approximate computations

### Attack Simulation

- **Model Poisoning**: Amplifies parameter updates
- **Gradient Reversal**: Trains to maximize loss
- **Label Flipping**: Corrupts training labels
- **Untargeted**: Random noise injection

## 🚀 Performance Optimization

### GPU Utilization

- Use appropriate CUDA architecture in CMake
- Monitor GPU memory usage
- Batch operations when possible

### Memory Management

- Clear intermediate tensors
- Use `with torch.no_grad():` for inference
- Implement checkpointing for large models

### Computation Optimization

- Enable lazy relinearization in CAHEL
- Use hoisting for rotation operations
- Optimize batch sizes for your hardware

## 🧪 Testing

### Unit Tests

Create tests for your components:

```python
def test_my_aggregator():
    aggregator = MyCustomAggregator()
    # Test with dummy models
    assert aggregator.aggregate(dummy_models) is not None
```

### Integration Tests

Test end-to-end workflows:

```python
def test_training_pipeline():
    args = create_test_args()
    trainer = LancelotTrainer(args)
    metrics = trainer.train()
    assert metrics['test_accuracies'][-1] > 0.5
```

## 📝 Contributing

### Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Add docstrings to all public methods
- Keep functions focused and modular

### Documentation

- Update this guide when adding features
- Include examples in docstrings
- Document performance implications
- Add references to relevant papers

### Pull Request Process

1. Create feature branch from main
2. Implement changes with tests
3. Update documentation
4. Submit PR with clear description

## 🔗 References

- [Original Paper](https://www.nature.com/articles/s42256-025-01107-6)
- [CAHEL Library Documentation](cahel-main/README.md)
- [OpenFHE Documentation](https://openfhe.org/)
- [PyTorch Federated Learning](https://pytorch.org/blog/federated-learning-with-pytorch/)

## 🆘 Common Issues

### CAHEL Build Issues

```bash
# Ensure correct GPU architecture
cmake -DCMAKE_CUDA_ARCHITECTURES=89  # For RTX 4090
cmake -DCMAKE_CUDA_ARCHITECTURES=75  # For T4
```

### Memory Issues

- Reduce batch size or number of clients
- Use gradient checkpointing
- Clear GPU cache: `torch.cuda.empty_cache()`

### Convergence Issues

- Check learning rate
- Verify data distribution
- Monitor Byzantine fraction
- Use different random seeds
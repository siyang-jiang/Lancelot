# Lancelot: Byzantine-Robust Federated Learning with Fully Homomorphic Encryption

[![Paper](https://img.shields.io/badge/Paper-Nature%20Machine%20Intelligence-blue)](https://www.nature.com/articles/s42256-025-01107-6)
[![License](https://img.shields.io/badge/License-MIT-green)]()

This repository contains the official implementation of **Lancelot**, a compute-efficient Byzantine-robust federated learning system with fully homomorphic encryption, as published in *Nature Machine Intelligence*.

## 📖 Overview

Lancelot addresses two critical challenges in federated learning:
1. **Byzantine Robustness**: Defending against malicious clients that may send corrupted model updates
2. **Privacy Protection**: Using fully homomorphic encryption to protect client data and model updates

The system leverages GPU acceleration through the **CAHEL** library for efficient homomorphic computations.

## 🏗️ Architecture

```
Lancelot/
├── cahel-main/           # HomoMul GPU Accelerator (CAHEL library)
│   ├── include/          # Header files for CAHEL
│   ├── src/              # Core CAHEL implementation
│   ├── python/           # Python bindings
│   └── examples/         # CAHEL usage examples
└── lancelot-main-GPU/    # Main Lancelot implementation
    ├── main.py           # Training entry point
    ├── src/              # Core aggregation and update logic
    ├── utils/            # Utilities (options, datasets, attacks, etc.)
    └── run_lancelot.sh   # Execution script
```

## 🚀 Quick Start

### Prerequisites

- **Hardware**: NVIDIA GPU with ≥24GB RAM (e.g., RTX 4090, Tesla V100)
- **Software**:
  - CMake ≥ 3.25.0
  - GCC ≥ 9.4.0
  - CUDA Toolkit
  - Python 3.8+
  - Conda (recommended)

### One-Click Setup and Execution

```bash
# Clone the repository
git clone https://github.com/siyang-jiang/Lancelot-Dev.git
cd Lancelot-Dev

# Run the complete setup and training pipeline
bash run.sh
```

**⚠️ Important**: Before running, update the GPU architecture in `run.sh` (line 32):
- For RTX 4090: `DCMAKE_CUDA_ARCHITECTURES=89`
- For Tesla T4: `DCMAKE_CUDA_ARCHITECTURES=75`
- For Tesla V100: `DCMAKE_CUDA_ARCHITECTURES=70`

## 🔧 Manual Installation

### Step 1: System Dependencies

```bash
# Update Ubuntu and install dependencies
sudo apt-get update
sudo apt-get install -y software-properties-common lsb-release
sudo apt-get clean all

# Add CMake repository
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
sudo apt-get update

# Install CMake
sudo dpkg --configure -a
sudo apt install cmake
```

### Step 2: Build CAHEL Library

```bash
cd cahel-main

# Deactivate conda if active
conda deactivate

# Configure and build
cmake -S. -Bbuild -DCMAKE_CUDA_ARCHITECTURES=89  # Adjust for your GPU
cmake --build build -j8
```

### Step 3: Setup Python Environment

```bash
cd build
conda activate your_environment  # Activate your preferred environment

# Install conda-build if not available
conda install conda-build

# Link the built library
conda develop lib
```

### Step 4: Install Python Dependencies

```bash
cd ../lancelot-main-GPU
pip install torch torchvision tqdm tensorboard numpy
```

## 🎯 Usage

### Basic Training

```bash
cd lancelot-main-GPU
bash run_lancelot.sh
```

### Custom Configuration

```bash
python main.py \
    --dataset CIFAR10 \
    --num_clients 10 \
    --c_frac 0.2 \
    --global_ep 100 \
    --method krum \
    --cipher_open 1
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset (CIFAR10, MNIST, FaMNIST) | CIFAR10 |
| `--num_clients` | Number of federated clients | 4 |
| `--c_frac` | Fraction of compromised clients | 0.0 |
| `--method` | Aggregation method (krum, trimmed_mean) | krum |
| `--cipher_open` | Enable homomorphic encryption | 0 |
| `--global_ep` | Number of communication rounds | 100 |
| `--lr` | Learning rate | 0.001 |

## 🛡️ Security Features

### Byzantine Attacks Supported
- **Untargeted attacks**: Random noise injection
- **Targeted attacks**: Model poisoning with specific objectives
- **Data poisoning**: Corrupted training data

### Defense Mechanisms
- **Krum**: Robust aggregation based on geometric median
- **Trimmed Mean**: Outlier removal and averaging
- **Homomorphic Encryption**: Privacy-preserving computation

## 📊 Performance

The system provides detailed timing analysis:
- Local training time
- Encryption/decryption overhead
- Aggregation computation time
- End-to-end communication rounds

Results are logged and can be visualized using TensorBoard:
```bash
tensorboard --logdir runs/
```

## 🔬 Research & Citation

If you use Lancelot in your research, please cite:

```bibtex
@article{jiang2025towards,
  title={Towards compute-efficient Byzantine-robust federated learning with fully homomorphic encryption},
  author={Jiang, Siyang and Yang, Hao and Xie, Qipeng and Ma, Chuan and Wang, Sen and Liu, Zhe and Xiang, Tao and Xing, Guoliang},
  journal={Nature Machine Intelligence},
  pages={1--12},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## 📋 Recent Updates

- **[2025-08-28]** 🔥 Paper accepted by Nature Machine Intelligence
- **[2025-08-20]** 🔥 Code Ocean capsule verified
- **[2025-08-15]** 🔥 Added end-to-end execution script (`run.sh`)
- **[2025-08-10]** 🔥 GPU acceleration optimizations

## 🤝 Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/siyang-jiang/Lancelot-Dev/issues)
- **Email**: Contact the authors for research collaboration
- **Code Ocean**: Verified capsule available (link coming soon)

## 🙏 Acknowledgments

- CAHEL library for GPU-accelerated homomorphic encryption
- OpenFHE library for comparison benchmarks
- The federated learning research community

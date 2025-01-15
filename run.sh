#!/usr/bin/env bash
set -ex

# This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.
# init conda
source ~/miniconda3/bin/activate
conda activate lancelot

# build and install openfhe
git clone https://github.com/openfheorg/openfhe-development.git
cd openfhe-development/
cmake -S. -Bbuild -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
cmake --build build -j4
cmake --install build
cd ..

## build openfhe python wrapper
git clone https://github.com/openfheorg/openfhe-python.git
cd openfhe-python/
cmake -S. -Bbuild
cmake --build build -j4
cmake --install build
mkdir lib
mv build/*.so lib
cd ../

# init openfhe wrapper
conda develop openfhe-python/lib

# build and install cahel
cd cahel-main/
cmake -S. -Bbuild -DCMAKE_CUDA_ARCHITECTURES=75 # specify your GPU architecture (e.g. 75 is for T4, 89 is for 4090)
cmake --build build -j4
cmake --install build
cd ../

# init cahel wrapper
conda develop cahel-main/build/lib






# run Lancelot
python -u  lancelot-main-GPU/main.py --gpu 0 --method krum --tsboard  --quantity_skew --global_ep 5 --cipher_open 1 --checks true --seed 2025 --dataset MNIST "$@"

python -u  lancelot-main-GPU/main.py --gpu 0 --method krum --tsboard  --quantity_skew --global_ep 5 --cipher_open 0 --seed 2025 --dataset MNIST "$@"


# run OpenFHE

python -u lancelot-main-GPU/main.py --gpu 0 --method krum --tsboard  --quantity_skew --global_ep 2 --cipher_open 1 --seed 2025 --dataset MNIST "$@"

python -u lancelot-main-GPU/main.py --gpu 0 --method krum --tsboard  --quantity_skew --global_ep 2 --cipher_open 0 --seed 2025 --openfhe true --dataset MNIST "$@"


# python -u lancelot-main-GPU/main.py "$@"



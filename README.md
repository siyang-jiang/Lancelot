#  Towards compute-efficient Byzantine-robust federated learning with fully homomorphic encryption
### Lancelot Offical Source Code 

This repo is the official repo for [Towards compute-efficient Byzantine-robust federated learning with fully homomorphic encryption](https://www.nature.com/articles/s42256-025-01107-6).

To evaluate Lancelot, we first install HomoMul GPU Accelerator (cahel) and then using our example code in lancelot-main-GPU.

> Device: NVIDIA GPU RAM > 24G (such as 4090)
> cmake version >= 3.25.0
> gcc version >= 9.4.0

## Update 
- [2025-08-28] Lancelot has been accepted by Nature Machine Intelligence. 🔥🔥🔥
- The Code Ocean capsule has passed code verification. 🔥🔥
- We will share the Code Ocean link soon. 🔥
- We have added an end-to-end script for running Lancelot (see run.sh). 🔥🔥

> Note that specify your GPU architecture (e.g. 75 is for T4, 89 is for 4090) in CMAKE (Line 32 in run.sh). 
```
 bash run.sh 
```


## Install HomoMul GPU Accelerator cahel-main

##### Update the Ubuntu
```
sudo apt-get update
sudo apt-get install -y software-properties-common lsb-release
sudo apt-get clean all
sudo apt-add-repository 'deb` `https://apt.kitware.com/ubuntu/ focal main'
```


##### Install the Cmake and gcc

```
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
sudo apt-get update
sudo dpkg --configure -a
sudo apt install cmake
```



##### Download the Lancelot

```
https://github.com/siyang-jiang/Lancelot-Dev.git
cd cahel-main
conda deactivate (If you open the conda environment)
cmake -S. -Bbuild -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j8
```
##### Conncet with python

```
cd build
conda activate
conda develop lib
```
(pleae make sure you have already install conda-build, otherwise)
```
conda install conda-build
```



## Evaluate Lancelot
```
cd lancelot-main-GPU
bash run_lancelot.sh
```

<!-- ## Results
![Results](https://github.com/siyang-jiang/Lancelot-Dev/blob/main/results.jpeg) -->

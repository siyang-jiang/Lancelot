# Lancelot-Dev

## Install cahel-main

##### Update the Ubuntu
`sudo apt-get update`
`sudo apt-get install -y software-properties-common lsb-
release`
`sudo apt-get clean all`
`sudo apt-add-repository 'deb` `https://apt.kitware.com/ubuntu/ focal main'`

##### Install the Cmake and gcc
(make sure your cmake version is above 3.25.0 and gcc version is above 9.4.0)

`wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -`

`sudo apt-get update`

`sudo dpkg --configure -a`

`sudo apt install cmake` 



##### Download the Lancelot
`https://github.com/siyang-jiang/Lancelot-Dev.git`

`cd cahel-main`

`Conda deactivate` (Close all conda environments) 

`cmake -S. -Bbuild -DCMAKE_CUDA_ARCHITECTURES=89`

`cmake --build build -j8`

`cd build`

`conda activate`

`conda develop lib`

#!/usr/bin/bash

#srun -n 8 -c 1 --gpus-per-node=8 --gpu-bind=closest ./proxy.sh
./proxy.sh


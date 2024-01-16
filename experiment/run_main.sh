#!/usr/bin/env bash
STARTING_GPU_INDEX=$1
GPU_NUM_PER_SERVER=$2
WORKER_NUM=$3

DATA_SET=$4
TASK_NUM=$5
MODEL=$6
CLIENT_NUM=$7
SAMPLE_NUM=$8
BATCH_SIZE=$9
LR=${10}

ALGO=${11}
EPSILON=${12}
EPS_INC=${13}

IID=${14}
SEED=${15}
TASK_PCT=${16}

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main.py \
  --starting_gpu $STARTING_GPU_INDEX \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --gpu_worker_num $WORKER_NUM \
  --dataset $DATA_SET \
  --num_task $TASK_NUM \
  --model $MODEL \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $SAMPLE_NUM \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --algo $ALGO \
  --epsilon $EPSILON \
  --eps_inc $EPS_INC \
  --iid $IID \
  --test_seed $SEED \
  --task_pct_per_client $TASK_PCT


  # mnist
  # sh run_main.sh 0 8 8  mnist_10ts    10 mlp            125 64 64 0.01   orth 0.96 0.0   0 0 1.0
  # sh run_main.sh 0 8 8  mnist_10ts    10 mlp            125 64 64 0.01   orth 0.94 0.0   1 0 1.0  

  # five-datasets
  # sh run_main.sh 0 8 8  five_datasets 5 resnet          150 64 16 0.01   orth 0.95 0.001 1 0 1.0

  # split-Cifar100
  # sh run_main.sh 0 8 8  splitCifar100 10 alexnet         50 25 16 0.01   orth 0.87 0.001 1 0 1.0

  # mini-imagenet
  # sh run_main.sh 0 8 8  imagenet 20 resnet_imagenet      50 25 16 0.01   orth 0.90 0.001 1 0 1.0







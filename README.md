## Introduction
This directory contains our jax implementation of Dcoformer. Note that our state-of-the-art results reported in the paper were obtained by training the model on a large-scale TPU cluster.

## Environment
python==3.10.10  
jax==0.4.25


### Data Prepare


### Train
#### Clone repositories
    git clone https://github.com/Caiyun-AI/DCFormer.git

#### Create tpu
    # v3 create command
    gcloud alpha compute tpus tpu-vm create $TPU_NAME --zone=$ZONE --accelerator-type=$TPU_TYPE --version=tpu-vm-base --project=$PROJECT_ID  --scopes=https://www.googleapis.com/auth/cloud-platform --preemptible

    # v4 create command
    gcloud alpha compute tpus tpu-vm create $TPU_NAME --zone=$ZONE --accelerator-type=$TPU_TYPE --version=tpu-vm-tf-2.10.0-pod-v4 --project=$PROJECT_ID  --scopes=https://www.googleapis.com/auth/cloud-platform --preemptible
  
    # v5p create command
    gcloud alpha compute tpus queued-resources create $TPU_NAME --node-id $TPU_NAME  --project $PROJECT_ID   --zone=$ZONE   --accelerator-type=$TPU_TYPE --runtime-version v2-alpha-tpuv5 --service-account $SERVICE_ACCOUNT   --best-effort

*SERVICE_ACCOUNT*: it can be obtained through command ‘gcloud iam service-accounts list’. The result is similar to: ***@developer.gserviceaccount.com  

*TPU_NAME*: tpu name  

*TPU_TYPE*: tpu type, such as: v3-8, v3-32, v4-8, v4-32, v5p-8, v5p-32.....  

*PROJECT_ID*: your project id  

*--preemptible/best-effort*: if you don't want to create a preemption, you can remove this parameter  

#### Install
    pip install -r MaxText/requirements_tpu.txt  # tpu
    pip install -r MaxText/requirements_gpu.txt  # gpu

#### Train on TPU
    export TPU_NAME=...
    export ZONE=...
    RUN_NAME='gs:/...'  # checkpoint and tensorboard save dir
    CONFIG_FILE=...  # dcformer_pp_405m.yml
    export HARDWARE=tpu # gpu or tpu

    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="export HARDWARE=tpu; python MaxText/train.py MaxText/configs/$CONFIG_FILE run_name=$RUN_NAME hardware=tpu |tee train.log"

#### Train on GPU
    RUN_NAME='gs:/...'  # checkpoint and tensorboard save dir
    CONFIG_FILE=...  # configs/*.yml
    export HARDWARE=gpu # gpu or tpu
    python MaxText/train.py MaxText/configs/$CONFIG_FILE run_name=$RUN_NAME hardware=gpu  compile_topology_num_slices=1 |tee train.log


### Experiments

- **405m dcformer++ vs transformer++**





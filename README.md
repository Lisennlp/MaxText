## Introduction
This directory contains our jax implementation of Dcoformer. Note that our state-of-the-art results reported in the paper were obtained by training the model on a large-scale TPU cluster.

In addition, the code of this library is modified based on [google/MaxText](https://github.com/google/maxtext) (thanks to google for its contribution), and most of its parameters and functions are used. At the same time, for the sake of simplicity of the code, some that are not used in our article are also removed Function. For the use of other functions, please refer to the source [MaxText](https://github.com/google/maxtext) library

## Environment
```plaintext
python==3.10.10  
jax==0.4.25
```

### Table of Contents
- [Getting Started](#Getting-started)
    - [1. Data Preparation](#1-data-preparation)
    - [2. Clone repositories](#2-clone-repositories)
    - [3. Create tpu](#3-Create-tpu)
    - [4. Install](#4-Install)
    - [5. Train on different hardware](#5-Train-on-different-hardware)
- [Use your dataset](#Use-your-dataset)
- [Tensorboard](#Tensorboard)
- [Experiments](#Experiments)



## Getting Started

#### 1. Data Preparation

#### 2. Clone Repositories
```bash
git clone https://github.com/Caiyun-AI/DCFormer.git
```
#### 3. Create TPU
 
```bash
# v3 create command
gcloud alpha compute tpus tpu-vm create $TPU_NAME --zone=$ZONE --accelerator-type=$TPU_TYPE --version=tpu-vm-base --project=$PROJECT_ID  --scopes=https://www.googleapis.com/auth/cloud-platform --preemptible

# v4 create command
gcloud alpha compute tpus tpu-vm create $TPU_NAME --zone=$ZONE --accelerator-type=$TPU_TYPE --version=tpu-vm-tf-2.10.0-pod-v4 --project=$PROJECT_ID  --scopes=https://www.googleapis.com/auth/cloud-platform --preemptible

# v5p create command
gcloud alpha compute tpus queued-resources create $TPU_NAME --node-id $TPU_NAME  --project $PROJECT_ID   --zone=$ZONE   --accelerator-type=$TPU_TYPE --runtime-version v2-alpha-tpuv5 --service-account $SERVICE_ACCOUNT   --best-effort
```
*```SERVICE_ACCOUNT```*: &nbsp;it can be obtained through command &nbsp; ```gcloud iam service-accounts list```. The result is similar to: ```***@developer.gserviceaccount.com```   
*```TPU_NAME```*:&nbsp;tpu name  
*```TPU_TYPE```*:&nbsp;tpu type, v3-8, v3-32, v4-8, v4-32, v5p-8, v5p-32 etc. 
*```PROJECT_ID```*: your project id  
*```--preemptible/best-effort```*:&nbsp;if you don't want to create a preemption, you can remove this parameter  

#### 4. Install

```bash
pip install -r MaxText/requirements_tpu.txt  # for tpu
pip install -r MaxText/requirements_gpu.txt   # for gpu
```


#### 5. Train On Different Hardware
- **Train on TPU**
```bash
TPU_NAME=...  # tpu name
ZONE=... # tpu zone
PIP_OR_PYTHON_PATH=...  # python or pip bin dir
WORKDIR=... # worddir
RUN_NAME=... # checkpoint and tensorboard save, it can be local dir or bucket dir(gs://...)
CONFIG_FILE=...  # configs/*.yml
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="export HARDWARE=tpu; cd $WORKDIR; $PIP_OR_PYTHON_PATH/python MaxText/train.py MaxText/configs/$CONFIG_FILE run_name=$RUN_NAME hardware=tpu | tee train.log"
```

- **Example on TPU**

```bash
TPU_NAME=my-tpu
ZONE=us-central1-a
PIP_OR_PYTHON_PATH=/home/xxx/miniconda3/bin
WORKDIR=gs://projects/DcFormer/MaxText # also use local dir
CONFIG_FILE=dcformer_pp_405m.yml
RUN_NAME=$WORKDIR/output/
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="$PIP_OR_PYTHON_PATH/pip install -r $WORKDIR/requirements_tpu.txt"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="export HARDWARE=tpu; cd $WORKDIR; $PIP_OR_PYTHON_PATH/python MaxText/train.py MaxText/configs/$CONFIG_FILE run_name=$RUN_NAME hardware=tpu | tee train.log"
```

- **Train on GPU**
```bash
PIP_OR_PYTHON_PATH=...  # python or pip bin dir
WORKDIR=... # workdir
RUN_NAME=...  # checkpoint and tensorboard save, it can be local dir or bucket dir(gs://...)
CONFIG_FILE=...  # configs/*.yml
export HARDWARE=gpu # gpu or tpu
$PIP_OR_PYTHON_PATH/python MaxText/train.py MaxText/configs/$CONFIG_FILE run_name=$RUN_NAME hardware=gpu  compile_topology_num_slices=1 | tee train.log
```

- **Example on GPU**
```bash
PIP_OR_PYTHON_PATH=/home/xxx/miniconda3/bin
WORKDIR=/home/xxx/projects/DcFormer/MaxText
CONFIG_FILE=dcformer_pp_405m.yml
RUN_NAME=$WORKDIR/output/
$PIP_OR_PYTHON_PATH/python MaxText/train.py MaxText/configs/$CONFIG_FILE run_name=$RUN_NAME hardware=gpu  compile_topology_num_slices=1 | tee train.log
```


## Use Your Dataset

You can change it to your own dataset by modifying the parameters ```dataset_path``` and ```dataset_type``` in the yml config file(default dataset is pile). In our library, we currently only support ```c4``` and ```pile``` datasets. If you need to use other self-processed or public datasets, you can Add the corresponding data processing files or functions to the file input_pipeline directory.

## Tensorboard

The train results include ```loss```、```grad```、```lr```etc message are writed tensorboard dir(default in $RUN_NAME/tensorboard). You can run a tensorboard program on local machine. such as:
    
```bash
tensorboad --logdir $RUN_NAME/tensorboard --bind_all --port 60000
```
You can view training-related information by visiting the URL （the IP + port of the machine you are running tensoboard on） after successful run
    

## Experiments

#### **405M dcformer++ vs transformer++**

![Loss曲线](images/405m_dcformer_pp_vs_transformer_pp_loss.png)




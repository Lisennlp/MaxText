## Introduction
This directory contains our jax implementation of Dcoformer. Note that our state-of-the-art results reported in the paper were obtained by training the model on a large-scale TPU cluster.

## Environment
```plaintext
python==3.10.10  
jax==0.4.25
```

### Data Prepare


### Getting Started
#### 1. Clone repositories
```bash
git clone https://github.com/Caiyun-AI/DCFormer.git
```
#### 2. Create tpu

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

#### 3. Install

```bash
pip install -r MaxText/requirements_tpu.txt  # for tpu
pip install -r MaxText/requirements_gpu.txt   # for gpu
```


#### 4 Train on different hardware
- Train on TPU
```bash
TPU_NAME=...  # tpu name
ZONE=... # tpu zone
PIP_OR_PYTHON_PATH=...  # python or pip bin dir
WORKDIR=/home/xxx/projects/MaxText # worddir
RUN_NAME=... # checkpoint and tensorboard save, it can be local dir or bucket dir(gs://...)
CONFIG_FILE=...  # configs/*.yml
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="export HARDWARE=tpu; cd $WORKDIR; $PIP_OR_PYTHON_PATH/python MaxText/train.py MaxText/configs/$CONFIG_FILE run_name=$RUN_NAME hardware=tpu | tee train.log"
```

- Train on GPU
```bash
PIP_OR_PYTHON_PATH=...  # python or pip bin dir
WORKDIR=/home/xxx/projects/MaxText # worddir
RUN_NAME=...  # checkpoint and tensorboard save, it can be local dir or bucket dir(gs://...)
CONFIG_FILE=...  # configs/*.yml
export HARDWARE=gpu # gpu or tpu
python MaxText/train.py MaxText/configs/$CONFIG_FILE run_name=$RUN_NAME hardware=gpu  compile_topology_num_slices=1 | tee train.log
```

- Example on TPU

```bash
TPU_NAME=my-tpu
ZONE=us-central1-a
PIP_OR_PYTHON_PATH=/home/xxx/miniconda3/bin
CONFIG_FILE=dcformer_pp_405m.yml
RUN_NAME=$WORKDIR/output/
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="$PIP_OR_PYTHON_PATH/pip install -r $WORKDIR/requirements_tpu.txt"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="export HARDWARE=tpu; cd $WORKDIR; $PIP_OR_PYTHON_PATH/python MaxText/train.py MaxText/configs/$CONFIG_FILE run_name=$RUN_NAME hardware=tpu | tee train.log"
```

- Tensorboard

The train results include ```loss```、```grad```、```lr```etc message are writed tensorboard dir(default in $RUN_NAME/tensorboard). You can run a tensorboard program on local machine. such as:
    
```bash
tensorboad --logdir $RUN_NAME/tensorboard --bind_all --port 60000
```
You can view training-related information by visiting the URL （the IP + port of the machine you are running tensoboard on） after successful run
    

### 5. Experiments

- **405m dcformer++ vs transformer++**

![Loss曲线](images/405m_dcformer_pp_vs_transformer_pp_loss.png)




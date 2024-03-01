## 创建 2 x v4-8 multi slice
export TPU_PREFIX=llm-jax-v4-8-multi
export QR_ID=$TPU_PREFIX
export NODE_COUNT=2
export ZONE=us-central2-b
RUN_NAME='multislice-test'
gcloud alpha compute tpus queued-resources create $QR_ID --accelerator-type=v4-8 --runtime-version=tpu-ubuntu2204-base --node-count=$NODE_COUNT --node-prefix=$TPU_PREFIX  --best-effort --zone $ZONE

## 进入tpu vm
gcloud compute tpus tpu-vm ssh $TPU_PREFIX-0 --zone=$ZONE --worker=0
gcloud compute tpus tpu-vm ssh $TPU_PREFIX-1 --zone=$ZONE --worker=0

## 安装环境
gcloud compute tpus tpu-vm scp requirements_lsp.txt $TPU_PREFIX-0:~/  --zone=$ZONE  --worker=all  --project=llm-tpu
gcloud compute tpus tpu-vm scp requirements_lsp.txt $TPU_PREFIX-1:~/  --zone=$ZONE  --worker=all  --project=llm-tpu

gcloud compute tpus tpu-vm ssh $TPU_PREFIX-0 --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/pip install -r requirements_lsp.txt"
gcloud compute tpus tpu-vm ssh $TPU_PREFIX-1 --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/pip install -r requirements_lsp.txt"

## 训练
### 远程训练
python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX --COMMAND="/home/lishengping/miniconda3/bin/python MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME" --ZONE $ZONE
### 进入vm
RUN_NAME='multislice-test'
python MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME



## v4-16 pod train single slice
## clone resposity
export TPU_PREFIX=llm-jax-v4-16-10
export QR_ID=$TPU_PREFIX
export ZONE=us-central2-b
RUN_NAME='gs://llm_base_models/lsp_test/singleslice'
gcloud compute tpus tpu-vm ssh $TPU_PREFIX --zone=$ZONE --worker=all --command="git clone https://github.com/Lisennlp/MaxText.git"
## 安装
gcloud compute tpus tpu-vm ssh $TPU_PREFIX --zone=$ZONE --worker=all --command="bash setup.sh;/home/lishengping/miniconda3/bin/pip install -r MaxText/requirements_lsp.txt"
gcloud compute tpus tpu-vm ssh $TPU_PREFIX --zone=$ZONE --worker=all --command="sudo lsof -w /dev/accel0 |cut -c 9-14|awk 'NR>1 {print $1}'| xargs sudo kill -9; sudo rm -f /tmp/libtpu_lockfile;sudo chmod +777 -R /tmp/tpu_logs/; killall main.py; cd MaxText; /home/lishengping/miniconda3/bin/python MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME"
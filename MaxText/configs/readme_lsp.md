## 创建
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


pip install -f https://storage.googleapis.com/jax-releases/libtpu_releases.html jax[tpu]==0.4.10
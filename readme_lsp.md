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
export TPU_PREFIX=llm-jax-v4-64-10
export QR_ID=$TPU_PREFIX
export ZONE=us-central2-b
<!-- RUN_NAME='gs://llm_base_models_us-central2/dcformer/maxtext/410m/qknorm0511/' -->
RUN_NAME='gs://llm_base_models_us-central2/dcformer/maxtext/405m/qknorm_qscale_use_w_false_v64_0512/'
RUN_NAME='gs://llm_base_models_us-central2/dcformer/maxtext/405m/test2/'

gcloud compute tpus tpu-vm ssh $TPU_PREFIX --zone=$ZONE --worker=all --command="sudo rm -r /home/lishengping/projects/MaxText; cd /home/lishengping/projects; git clone https://github.com/Lisennlp/MaxText.git"
## 安装
gcloud compute tpus tpu-vm ssh $TPU_PREFIX --zone=$ZONE --worker=all --command="cd /home/lishengping/projects/;/home/lishengping/miniconda3/bin/pip install -r MaxText/requirements_tpu.txt"
gcloud compute tpus tpu-vm ssh $TPU_PREFIX --zone=$ZONE --worker=all --command="sudo lsof -w /dev/accel0 |cut -c 9-14|awk 'NR>1 {print $1}'| xargs sudo kill -9; sudo rm -f /tmp/libtpu_lockfile;sudo chmod +777 -R /tmp/tpu_logs/; killall main.py; export HARDWARE='tpu'cd /home/lishengping/projects/MaxText; /home/lishengping/miniconda3/bin/python MaxText/train.py MaxText/configs/dcformer_pp_405m.yml  run_name=$RUN_NAME  > train.log 2>&1 &"


export TPU_PREFIX=llm-jax-v3-8-10
export ZONE=us-central1-a
RUN_NAME='gs://llm_base_models_us-east5/lsp_test/maxtext0419'
python MaxText/train.py MaxText/configs/410m_dcformer.yml run_name=$RUN_NAME |tee train.log


## v5p-64 pod train single slice
## clone resposity
export TPU_PREFIX=llm-jax-v5p-64-10
export QR_ID=$TPU_PREFIX
export ZONE=us-east5-a
RUN_NAME='gs://llm_base_models_us-east5/dcformer/maxtext/410m/'
gcloud compute tpus tpu-vm ssh $TPU_PREFIX --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/pip install -r /home/lishengping/projects/MaxText/requirements_lsp.txt"
gcloud compute tpus tpu-vm ssh $TPU_PREFIX --zone=$ZONE --worker=all --command=" sudo rm -r /home/lishengping/projects/MaxText; cd /home/lishengping/projects/; git clone https://github.com/Lisennlp/MaxText.git"

gcloud compute tpus tpu-vm ssh $TPU_PREFIX --zone=$ZONE --worker=all --command="sudo lsof -w /dev/accel0 |cut -c 9-14|awk 'NR>1 {print $1}'| xargs sudo kill -9; sudo rm -f /tmp/libtpu_lockfile;sudo chmod +777 -R /tmp/tpu_logs/; killall main.py; cd /home/lishengping/projects/MaxText; /home/lishengping/miniconda3/bin/python MaxText/train.py MaxText/configs/410m_dcformer.yml run_name=$RUN_NAM > train.log 2>&1 &"

GPU:
gcloud auth login
gcloud auth application-default login
export HARDWARE='gpu'
RUN_NAME='gs://llm_base_models_us-east5/lsp_test/maxtext/gpu/'
CUDA_VISIBLE_DEVICES=6,7 python MaxText/train.py MaxText/configs/dcformer_pp_405m.yml run_name=$RUN_NAME |tee train.log



TPU_NAME=llm-jax-v3-8-10
ZONE=us-central1-a
PIP_OR_PYTHON_PATH=/home/lishengping/miniconda3/bin
WORKDIR=/home/lishengping/projects/MaxText
CONFIG_FILE=test.yml
DATASET_PATH=gs://common_datasets_us-central2/pythia_pile_idxmaps_tfrecord/
RUN_NAME=$WORKDIR/410m/

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="$PIP_OR_PYTHON_PATH/pip install -r $WORKDIR/requirements_tpu.txt"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="export HARDWARE=tpu; cd $WORKDIR; $PIP_OR_PYTHON_PATH/python MaxText/train.py MaxText/configs/$CONFIG_FILE run_name=$RUN_NAME hardware=tpu dataset_path=$DATASET_PATH| tee train.log"




# 上传数据到huggingface过程
pip install huggingface_hub
git lfs install  or sudo apt install git-lfs
huggingface-cli login
git clone https://huggingface.co/datasets/Caiyun-AI/Pile_tfrecord
cd Pile_tfrecord
# 设置大文件传输和http传输上限
huggingface-cli lfs-enable-largefiles .
git config http.postBuffer 15728640000
# 将大文件的后缀或者格式加入到.gitattribute里面
git lfs track pile*

git add FILE_NAME
# 这一步可能会让你设置用户名和邮箱之类的，按照提示做就行了
git commit -m '提交信息'
git push

# 输入用户名密码之后 1小时内不用输入
git config --global credential.helper 'cache --timeout=3600'

Name: Caiyun-AI
Email: caiyunai2024@163.com
token: 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

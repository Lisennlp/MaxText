%load_ext autoreload
%autoreload 2

    
import os
import sys
import yaml
import json
import base64
from collections import defaultdict
from typing import Tuple
import functools

sys.path.append('/home/lishengping/projects/MaxText/MaxText')
os.environ['HARDWARE'] = 'tpu'

from layers import models
import max_utils
import jax
import orbax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax.traverse_util import flatten_dict, unflatten_dict
from flax import linen as nn
from transformers import AutoTokenizer
from etils import epath

import pyconfig
from jax.sharding import PartitionSpec
from flax.linen import partitioning as nn_partitioning


# TOKENIZER_PATH = '/home/lishengping/tokenizer'
# if not os.path.exists(TOKENIZER_PATH):
#     !gsutil cp -r gs://llm_base_models_us-east5/qwen/tokenizer /home/lishengping/
# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True, trust_remote_code=True)

read_dir = 'gs://llm_base_models_us-central2/dcformer/maxtext/410m/qknorm0511_scale/checkpoints'
read_dir = epath.Path(read_dir)

config_name = '/home/lishengping/projects/MaxText/MaxText/configs/410m_dcformer.yml'
# config_name = '/home/lishengping/projects/MaxText/MaxText/configs/dc_7b.yml'

argv = [None, config_name]
pyconfig.initialize(argv)
config = pyconfig.config
# validate_train_config(config)
devices_array = max_utils.create_device_mesh(config)
mesh = Mesh(devices_array, config.mesh_axes)


def decode_base64(encoded_str):
    decoded_bytes = base64.b64decode(encoded_str)
    decoded_str = decoded_bytes.decode('utf-8')
    return decoded_str


def mesh_shard_rules(mesh, rules, remove_keys=[]):
    _sharding_dict = {}
    for name, rule in rules.items():
        if isinstance(rule, str):
            rule = json.loads(rule)
        name = decode_base64(name)
        param_key = tuple(name.split('.'))
        remove = any([1 if key in param_key else 0 for key in remove_keys])
        if remove: continue
        prule = [tuple(r) if isinstance(r, list) else r for r in rule['partition_spec'] ]
        spec = jax.sharding.PartitionSpec(*prule)
        _sharding_dict[param_key] = jax.sharding.NamedSharding(mesh, spec)
    return _sharding_dict


def rewrite_bucket_sharding(mesh, old_sharding, save_path):
    cur_machine_sharding = {}
    for k, v in old_sharding.items():
        if isinstance(v, str):
            v = json.loads(v)
        v['shape'] = mesh.device_ids.shape
        cur_machine_sharding[k] = v
    save_path = epath.Path(save_path)
    with save_path.open('w') as f:
        json.dump(cur_machine_sharding, f)
    
load_step = 440000
_sharding_path = read_dir / '2000' / 'default/_sharding'
_metadata_path = read_dir / '2000' / 'default/_METADATA'

# delete file or dir
# _sharding_path.unlink()

remove_keys = ['opt_state', 'step']
if _sharding_path.exists():
    with _sharding_path.open('r') as f:
        _sharding_rules = json.load(f)
    # 重写_sharding文件
    # rewrite_bucket_sharding(mesh, _sharding_rules, _sharding_path)
    _sharding_dict = mesh_shard_rules(mesh, _sharding_rules, remove_keys=remove_keys)
    _sharding_dict = unflatten_dict(_sharding_dict)
elif _metadata_path.exists():
    _metadata_dict = {}
    with _metadata_path.open('r') as f:
        _metadata = json.load(f)
    for param_key in _metadata['tree_metadata']:
        if isinstance(param_key, str): param_key = eval(param_key)
        remove = any([1 if key in param_key else 0 for key in remove_keys])
        if remove: continue
        _metadata_dict[param_key] = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)
    _metadata_dict = unflatten_dict(_metadata_dict)
    
else:
    _sharding_dict = None
    _metadata_dict = None


checkpoint_dir = 'gs://llm_base_models_us-central2/dcformer/maxtext/410m/qknorm0511_scale_test/checkpoints/'
options = orbax.checkpoint.CheckpointManagerOptions()
item = {
  "default": orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler(use_ocdbt=True))
}
ocdbt_max_mngr = orbax.checkpoint.CheckpointManager(checkpoint_dir, item, options)
checkpoint_step = 13000
state = ocdbt_max_mngr.restore(checkpoint_step, items=item)
params = state['default']['params']


assert _sharding_dict is not None

@functools.partial(jax.jit, in_shardings=None, out_shardings=_sharding_dict['params'])
def shard_to_tpu(x):
    return x
tpu_params = shard_to_tpu(params)
flat_params = flatten_dict(tpu_params)
for k, v in flat_params.items():
    print(k, v.shape)
print(f'devices: {v.devices()}')

quant = None

Transformer = models.Transformer
model = Transformer(config, mesh, quant=quant)
is_train = False
rng1, aqt_rng = jax.random.split(jax.random.key(9876))



import os
import time
import argparse
import socket
import random
from collections import defaultdict

import tensorflow as tf
import jax
import numpy as np

import math
from typing import Dict, List, Optional

from google.cloud import storage


seq_len = 2049

def extract_v3p5_longdata_files(dataset_path):  # lsp
    random.seed(9876)
    client = storage.Client()
    #v3: us-east1-d -> common_datasets, v4: us-central2-b -> common_datasets_us-central2-b
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    train_files, valid_files = [], []
    train_long_files, train_short_files = [], []
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        if 'valid' in path:
            valid_files.append(path)
        else:
            if '.long' in path:
                train_long_files.append(path)
            else:
                train_short_files.append(path)
    # file size short：long = 1.5: 1, 为了保证short的token: long = 3: 7, 因此 short 取 (1 / 1.5) * (3 / 7) = 2 / 7
    short_k = min(3 * len(train_long_files) // 14, len(train_short_files))
    selected_short_files = random.sample(train_short_files, k=short_k)
    train_files = selected_short_files + train_long_files
    print(f'selected_short_files: {len(selected_short_files)} train_long_files: {len(train_long_files)}')
    random.shuffle(train_files)
    print(f'first 10 train files: {train_files[:10]}')
    valid_files = sorted(valid_files)
    print(f'valid_files: {valid_files}')
    return train_files, valid_files

def extract_pythia_datapath(dataset_path, eval_split):
    if not dataset_path:
      return []
    client = storage.Client()
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    step_map_path = {}
    eval_pathes = []
    rerank = 0
    print(f'bucket_name: {bucket_name} directory_path: {directory_path}')
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        print(f'blob.name: {blob.name}')
        if ".tfrecord" not in blob.name: continue
        try:
            step = int(blob.name.rsplit("pile.tfrecord.b", maxsplit=1)[-1])
        except:
            step = rerank
            rerank += 1
        path = f'gs://{os.path.join(bucket_name, blob.name)}'

        if eval_split in path:
            eval_pathes.append(path)
            continue
        step_map_path[step] = path

    # sorted_step_path = sorted(step_map_path.items(), key=lambda x: x[0])
    # print(sorted_step_path)
    # # steps, pathes = zip(*sorted_step_path)
    # if not isinstance(pathes, list):
    #     pathes = list(pathes)
    pathes = []
    print(f'pathes: {len(pathes)} eval_pathes: {len(eval_pathes)}')
    return pathes, eval_pathes
    
def extract_v3p5_data_files(dataset_path):
    client = storage.Client()
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    # logging.info(f'bucket_name = {bucket_name}, directory_path = {directory_path}')
    train_files, valid_files = [], []
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        if 'valid' in path:
            valid_files.append(path)
        else:
            train_files.append(path)
    train_files = sorted(train_files)
    # valid_files = sorted(valid_files)
    print(f'Train file: {len(train_files)},  test file: {len(valid_files)}')
    return train_files, valid_files
    

def _parse_function(example_proto):
    feature_desc = {key: tf.io.VarLenFeature(tf.int64) for key in task_features}
    example = tf.io.parse_single_example(example_proto, feature_desc)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        example[name] = tf.sparse.to_dense(t, default_value=0)[: seq_len]
        print(f'example[name]: {example[name]}')
    return example

task_features = {'input_ids': None}
train_seed = 1234
num_infeed_hosts = 1
shuffle_buffer_size = None
pad_id = 0
batch_size = 4

fname = ['gs://jax_llm_data/xiaomeng/sft_target/tfrecord_len2k/en.test.continue_write.tfrecord']
datadir = 'gs://jax_llm_data_us-east5/xiaomeng/v3.5/tfids_4k_32k_0622/valid_tfrecord'
# train_files, eval_files = extract_v3p5_longdata_files(datadir)

datadir = 'gs://jax_llm_data_us-east5/xiaomeng/v3.5/tfids0527'

datadir = 'gs://common_datasets_us-east5/pythia_model_test/pile_test/'

# train_files, eval_files = extract_v3p5_data_files(datadir)
train_files, eval_files = extract_pythia_datapath(datadir, 'val_with_eos')


fname = eval_files

# fname = ['gs://jax_llm_data/xiaomeng/sft_target/tfrecord_len2k/en.test.continue_write.tfrecord']
tf.random.set_seed(train_seed)
ds = tf.data.Dataset.from_tensor_slices(fname)
ds = ds.apply(tf.data.TFRecordDataset)
# shard host data
ds = ds.shard(num_infeed_hosts, 0)
ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
if shuffle_buffer_size is not None:
    ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)
padded_shapes = {key: seq_len for key in task_features}
padding_values = {key: pad_id for key in task_features}
ds = ds.padded_batch(
    batch_size=np.prod(batch_size),
    padded_shapes=padded_shapes,
    padding_values=padding_values,
    drop_remainder=True,
)
# ds = ds.map(self.convert)
# ds = ds.prefetch(tf.data.AUTOTUNE)
iter_ds = ds.as_numpy_iterator()


# x = next(iter_ds)
# # input_ids = jnp.array([[ 4678, 16741,   310,   253,  5347,   273]])
# input_ids = x['input_ids'][:, :129]
# input_ids = (input_ids>=50256) * 209 + (input_ids<50256) * input_ids

# print(f'input_ids: {input_ids.shape}')
# data = {}
# data['inputs'] = input_ids1[:, :-1]
# pos = jnp.arange(data['inputs'].shape[1]).reshape(1, -1)
# data["inputs_position"] = jnp.broadcast_to(pos, (batch_size, pos.shape[-1]))
# data["inputs_segmentation"] = jnp.ones_like(data['inputs'])
# data["targets"] = input_ids1[:, 1:]
# data = {k: v[:, :] for k, v in data.items()}

# logits, intermediate_outputs = model.apply(
#           {'params': tpu_params},
#           data["inputs"],
#           data["inputs_position"],
#           decoder_segment_ids=data["inputs_segmentation"],
#           enable_dropout=config.enable_dropout if is_train else False,
#           rngs={"dropout": rng1, "params": aqt_rng},
#           mutable="intermediates",
#       )
# one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
# xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)




def build_data_sharding(features, shard_names):
    shard_names = ('fsdp', None)
    data_sharding = {}
    for k in features:
        spec = jax.sharding.PartitionSpec(*shard_names)
        data_sharding[k] = jax.sharding.NamedSharding(mesh, spec)
    return data_sharding

data_features = ['inputs', 'inputs_position', 'inputs_segmentation', 'targets']
data_shard_names = ('data', None)
data_sharding = build_data_sharding(data_features, data_shard_names)

@functools.partial(jax.jit, in_shardings=(data_sharding, {'params': _sharding_dict['params']}, ), out_shardings=None)
def model_forward(data, params):
    logits, intermediate_outputs = model.apply(
          params,
          data["inputs"],
          data["inputs_position"],
          decoder_segment_ids=data["inputs_segmentation"],
          enable_dropout=config.enable_dropout if is_train else False,
          rngs={"dropout": rng1, "params": aqt_rng},
          mutable="intermediates",
      )
    one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
    xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
    xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
    return xent, intermediate_outputs  


x = next(iter_ds)
input_ids = x['input_ids'][:, :129]
print(f'input_ids: {input_ids.shape}')
data = {}
data['inputs'] = input_ids[:, :-1]
pos = jnp.arange(data['inputs'].shape[1]).reshape(1, -1)
data["inputs_position"] = jnp.broadcast_to(pos, (batch_size, pos.shape[-1]))
data["inputs_segmentation"] = jnp.ones_like(data['inputs'])
data["targets"] = input_ids[:, 1:]
data = {k: v[:, :] for k, v in data.items()}

# loss compute
loss, intermediate_outputs = model_forward(data, {'params': tpu_params})
print(f'loss shape: {loss.shape} mean: {loss.mean()}')
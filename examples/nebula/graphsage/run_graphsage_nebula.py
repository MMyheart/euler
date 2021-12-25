# coding=utf-8
# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import tf_euler
import os
import json
 
from euler_estimator import NodeEstimator
from graphsage import SupervisedGraphSage
 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
 
 
def define_network_flags():
    tf.flags.DEFINE_string('model', 'graphsage', 'model')
 
    tf.flags.DEFINE_integer('task_index', 0, '')
    tf.flags.DEFINE_list('ps_hosts', None, 'ps hosts')
    tf.flags.DEFINE_list('worker_hosts', None, 'worker hosts')
    tf.flags.DEFINE_enum('job_name', 'ps', ['ps', 'worker'], 'ps or worker')
 
    tf.flags.DEFINE_integer('hidden_dim', 32, 'Hidden dimension.')
    tf.flags.DEFINE_integer('layers', 2, 'SAGE convolution layer number.')
    tf.flags.DEFINE_list('fanouts', [10, 10], 'GraphSage fanouts.')
    tf.flags.DEFINE_integer('batch_size', 3200, 'Mini-batch size')
    tf.flags.DEFINE_integer('num_epochs', 1, 'Epochs to train')
    tf.flags.DEFINE_integer('log_steps', 1, 'Number of steps to print log.')
    tf.flags.DEFINE_string('model_dir', 'ckpt', 'Model checkpoint.')
    tf.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
    tf.flags.DEFINE_enum('optimizer', 'adam',
                         ['adam', 'adagrad', 'sgd', 'momentum', 'hashadam', 'hashadagrad', 'hashftrl'],
                         'Optimizer algorithm')
    tf.flags.DEFINE_enum('run_mode', 'train', ['train', 'evaluate', 'infer'], 'Run mode.')
 
    #
    tf.flags.DEFINE_enum('use_hash_embedding', 'false', ['true', 'false'], '是否使用 hash embedding')
    tf.flags.DEFINE_integer('hash_partition', None, '将 hash embedding 分为 hash_partition 片')
    tf.flags.DEFINE_enum('use_id', 'false', ['true', 'false'], '是否使用随机初始化的向量作为feature')
    tf.flags.DEFINE_string('log_dir', 'board', 'board log.')
 
    # infer
    tf.flags.DEFINE_string('infer_dir', 'infer', 'infer.')
    tf.flags.DEFINE_string('infer_id_dir', '', 'id文件目录')
    tf.flags.DEFINE_string('infer_out_dir', '', 'id文件目录')
    tf.flags.DEFINE_string('infer_mode', 'user', '')
 
    #
    tf.flags.DEFINE_integer('max_id', -1, 'max id')
    tf.flags.DEFINE_list('train_node_type', ['train'], 'train node type')
    tf.flags.DEFINE_list('train_edge_type', ['train'], 'train edge type')
    tf.flags.DEFINE_integer('total_size', 231443, '')
    tf.flags.DEFINE_list('all_node_types', ['test', 'train', 'val'], 'all node types')
    tf.flags.DEFINE_list('all_edge_types', ['train', 'train_removed'], 'all edges types')
    tf.flags.DEFINE_string('id_file', '', '')
    tf.flags.DEFINE_string('feature_idx', 'feature', '')
    tf.flags.DEFINE_integer('feature_dim', 602, '')
    tf.flags.DEFINE_string('label_idx', 'label', '')
    tf.flags.DEFINE_integer('label_dim', 41, '')
    tf.flags.DEFINE_integer('num_classes', 41, '')
 
    # nebula
    tf.flags.DEFINE_list('nebula_hosts', [],
                     'nebula graph cluster hosts.')
    tf.flags.DEFINE_integer('nebula_port', 8419, 'nebula graph cluster port.')
    tf.flags.DEFINE_integer('nebula_timeout', 60000, 'nebula client query timeout.')
    tf.flags.DEFINE_integer('nebula_pool_max', 30, 'nebula client pool max total.')
    tf.flags.DEFINE_integer('nebula_pool_min', 30, 'nebula client pool min total.')
    tf.flags.DEFINE_string('nebula_space', '', 'nebula space name.')
    tf.flags.DEFINE_integer('nebula_op_thread_num', 20, 'nebula op thead num.')
    tf.flags.DEFINE_list('nebula_ops', ['sample_neighbor', 'get_dense_feature', 'sample_node'],
                       'nebula op list') 
 
def initialize_nebula(flags_obj):
    tf_euler.initialize_nebula({
        'host': flags_obj.nebula_hosts[0],
        'hosts': flags_obj.nebula_hosts,
        'port': flags_obj.nebula_port,
        'pool_size': 1,
        'timeout': flags_obj.nebula_timeout,
        'minConnectionNum': flags_obj.nebula_pool_max,
        'maxConnectionNum': flags_obj.nebula_pool_min,
        'user': 'user',
        'password': 'password',
        'space': flags_obj.nebula_space,
        'all_node_types': flags_obj.all_node_types,
        'all_edge_types': flags_obj.all_edge_types,
        'ops': flags_obj.nebula_ops,
        'op_thread_num': flags_obj.nebula_op_thread_num
    })
 
 
def run_model(flags_obj):
    fanouts = list(map(int, flags_obj.fanouts))
    assert flags_obj.layers == len(fanouts)
    dims = [flags_obj.hidden_dim] * (flags_obj.layers + 1)
    if flags_obj.run_mode == 'train':
        metapath = [flags_obj.train_edge_type] * flags_obj.layers
    else:
        metapath = [flags_obj.all_edge_types] * flags_obj.layers
    num_steps = int((flags_obj.total_size + 1) // flags_obj.batch_size *
                    flags_obj.num_epochs)

    model = SupervisedGraphSage(dims, fanouts, metapath,
                                flags_obj.feature_idx,
                                flags_obj.feature_dim,
                                flags_obj.label_idx,
                                flags_obj.label_dim)
 
    params = {'train_node_type': flags_obj.train_node_type[0],
              'batch_size': flags_obj.batch_size,
              'optimizer': flags_obj.optimizer,
              'learning_rate': flags_obj.learning_rate,
              'log_steps': flags_obj.log_steps,
              'model_dir': flags_obj.model_dir,
              'id_file': flags_obj.id_file,
              'infer_dir': flags_obj.infer_dir,
              'total_step': num_steps,
              'log_dir': flags_obj.log_dir}
 
    config = tf.estimator.RunConfig(log_step_count_steps=None)
    model_estimator = NodeEstimator(model, params, config)
 
    # run mode
    if flags_obj.run_mode == 'train':
        model_estimator.train()
    elif flags_obj.run_mode == 'evaluate':
        model_estimator.evaluate()
    elif flags_obj.run_mode == 'infer':
        model_estimator.infer()
    else:
        raise ValueError('Run mode not exist!')
 
 
def run_distributed(flags_obj):
    if flags_obj.ps_hosts is None:
        cluster_config = {
            'chief': flags_obj.worker_hosts[0:1],
            'worker': flags_obj.worker_hosts[1:]
        }
    else:
        cluster_config = {
            'chief': flags_obj.worker_hosts[0:1],
            'ps': flags_obj.ps_hosts,
            'worker': flags_obj.worker_hosts[1:]
        }
    cluster = tf.train.ClusterSpec(cluster_config)
    # handle ps
    if flags_obj.job_name == 'ps' and flags_obj.ps_hosts is not None:
        if flags_obj.run_mode == 'infer':
            initialize_nebula(flags_obj)
        server = tf.train.Server(
            cluster, job_name=flags_obj.job_name, task_index=flags_obj.task_index)
        server.join()
    # handle worker
    elif flags_obj.job_name == 'worker':
        task_index = flags_obj.task_index
        job_type = flags_obj.job_name if task_index != 0 else 'chief'
        task_index = task_index if task_index == 0 else task_index - 1
        server = tf.train.Server(
            cluster, job_name=job_type, task_index=task_index)
        with tf.device(
                tf.train.replica_device_setter(
                    worker_device='/job:{}/task:{}'.format(job_type, task_index),
                    cluster=cluster)):
            os.environ['TF_CONFIG'] = json.dumps({
                'cluster': cluster_config,
                'task': {
                    'type': job_type,
                    'index': task_index
                }
            })
            initialize_nebula(flags_obj)
            run_model(flags_obj)
    else:
        raise ValueError('Unsupported job name: {}'.format(flags_obj.job_name))
 
 
def run_local(flags_obj):
    initialize_nebula(flags_obj)
    run_model(flags_obj)
 
 
def main(_):
    # get init params
    flags_obj = tf.flags.FLAGS
    # init use_hash_embedding
    use_hash_embedding = flags_obj.use_hash_embedding
    if use_hash_embedding == 'true':
        print("[train] os.putenv('use_hash_embedding', 'true')")
        os.environ['use_hash_embedding'] = 'true'
    #
    if flags_obj.worker_hosts is None or (len(flags_obj.worker_hosts) == 1 and flags_obj.ps_hosts is None):
        run_local(flags_obj)
    else:
        run_distributed(flags_obj)
 
 
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_network_flags()
    tf.app.run(main)

# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import os

import tensorflow as tf
import logging

from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_uint
 
nebula_space = None
nebula_all_edge_types = None
nebula_all_node_types = None
nebula_ops = {
    'get_full_neighbor': False,
    'sample_neighbor': False,
    'sample_neighbor_layerwise': False,
    'get_dense_feature': False,
    'get_sparse_feature': False,
    'get_edge_sparse_feature': False,
    'get_edge_dense_feature': False,
    'get_binary_feature': False,
    'get_edge_binary_feature': False,
    'get_graph_by_label': False,
    'sample_fanout': False,
    'sample_fanout_with_feature': False,
    'sample_node': False,
    'sample_edge': False,
    'random_walk': False,
    'sparse_get_adj': False,
    'get_node_type_id': False,
    'get_edge_type_id': False,
    'get_multi_hop_neighbor': False,
    'get_sorted_full_neighbor': False,
    'get_top_k_neighbor': False,
    'sample_graph_label': False,
    'sample_node_with_src': False,
}
nebula_ops_refer = {
    'sample_fanout_layerwise_each_node': [
        'get_edge_type_id',
        'sample_neighbor',
        'sample_neighbor_layerwise',
    ],
    'sample_fanout_layerwise': [
        'get_edge_type_id',
        'sample_neighbor_layerwise',
    ],
    'get_multi_hop_neighbor': [
        'get_edge_type_id',
        'get_full_neighbor',
    ],
}
nebula_op_thread_num = 20
nebula_graph_label_start = 0
nebula_graph_label_end = 0


logger = logging.getLogger("simple_example")
logger.setLevel(logging.DEBUG)

_LIB_DIR = os.path.dirname(os.path.realpath(__file__))
_LIB_PATH = os.path.join(_LIB_DIR, 'libtf_euler.so')

_LIB_OP = tf.load_op_library(_LIB_PATH)
_LIB = ctypes.CDLL(_LIB_PATH)


def initialize_nebula(config):
    global nebula_space
    global nebula_all_edge_types
    global nebula_all_node_types
    global nebula_ops
    global nebula_graph_label_start
    global nebula_graph_label_end
    global nebula_op_thread_num
    nebula_space = config['space']
    nebula_all_edge_types = config['all_edge_types']
    nebula_all_node_types = config['all_node_types']
    if 'graph_label_start' in config:
        nebula_graph_label_start = config['graph_label_start']
    if 'graph_label_end' in config:
        nebula_graph_label_end = config['graph_label_end']
    if 'ops' in config:
        for op in config['ops']:
            set_nebula_op(op, True)
    if 'op_thread_num' in config:
        nebula_op_thread_num = config['op_thread_num']
 
    # init nebula graph
    if 'port' in config and 'hosts' in config and 'timeout' in config \
        and 'minConnectionNum' in config and 'maxConnectionNum' in config :
        port = config['port']
        hosts = ','.join(config['hosts'])
        timeout = config['timeout']
        minConnectionNum = config['minConnectionNum']
        maxConnectionNum = config['maxConnectionNum']
        initNebulaGraph = _LIB.InitNebulaGraph
        initNebulaGraph.argtypes=[c_char_p, c_uint, c_int, c_uint, c_uint]
        initNebula = initNebulaGraph(hosts, port, timeout, minConnectionNum, maxConnectionNum)
        print("InitNebulaGraph " + str(initNebula))
 
def set_nebula_op(op, is_enable):
    nebula_ops[op] = is_enable
    if op in nebula_ops_refer:
        for refer_func in nebula_ops_refer[op]:
            nebula_ops[refer_func] = is_enable
 

def initialize_graph(config):
    """
    Initialize the Euler graph driver used in Tensorflow.

    Args:
      config: str or dict of Euler graph driver configuration.

    Return:
      A boolean indicate whether the graph driver is initialized successfully.

    Raises:
      TypeError: if config is neither str nor dict.
    """
    if isinstance(config, dict):
        config = ';'.join(
            '{}={}'.format(key, value) for key, value in config.items())
    if not isinstance(config, str):
        raise TypeError('Expect str or dict for graph config, '
                        'got {}.'.format(type(config).__name__))

    if not isinstance(config, bytes):
        config = config.encode()

    return _LIB.InitQueryProxy(config)


def initialize_embedded_graph(data_dir, sampler_type='all', data_type='all'):
    return initialize_graph({'mode': 'local',
                             'data_path': data_dir,
                             'data_type': data_type,
                             'sampler_type': sampler_type})


def initialize_shared_graph(data_dir, zk_addr, zk_path, shard_num):
    return initialize_graph({'mode': 'remote',
                             'zk_server': zk_addr,
                             'zk_path': zk_path,
                             'shard_num': shard_num,
                             'num_retries': 1})

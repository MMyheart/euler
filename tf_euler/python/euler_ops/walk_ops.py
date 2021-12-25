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

import tensorflow as tf

from tf_euler.python.euler_ops import base
from tf_euler.python.euler_ops import type_ops
import numpy as np

gen_pair = base._LIB_OP.gen_pair
_random_walk = base._LIB_OP.random_walk


def random_walk(nodes, edge_types, p=1.0, q=1.0, default_node=-1):
    '''
    Random walk from a list of nodes.

    Args:
    nodes: start node ids, 1-d Tensor
    edge_types: list of 1-d Tensor of edge types
    p: back probality
    q: forward probality
    default_node: default fill nodes
    '''

    if base.nebula_ops['random_walk']:
        return nebula_random_walk(nodes, edge_types, p, q, default_node)
    edge_types = [type_ops.get_edge_type_id(edge_type)
                  for edge_type in edge_types]
    return _random_walk(nodes, edge_types, p, q, default_node)

def nebula_random_walk(nodes, edge_types, p=1.0, q=1.0, default_node=-1):
    result = tf.py_func(
        _nebula_random_walk,
        [nodes, edge_types, p, q, default_node],
        [tf.int64],
        True,
        'NebulaRandomWalk'
    )
    result[0].set_shape((nodes.shape.dims[0].value, len(edge_types) + 1))
    return result[0]
 
 
def _nebula_random_walk(nodes, edge_types, p, q, default_node):
    paths = []
    uniq_nodes = {}.fromkeys(nodes).keys()
    nql = 'USE {}; randomwalk {} from {} over {} where p=={} and q=={}'.format(
        base.nebula_space,
        len(edge_types),
        ', '.join(str(x) for x in uniq_nodes),
        ', '.join(str('e_' + x) for x in edge_types[0]),
        p,
        q
    )
    path_cache = {}
    resp = base.nebula_client.execute_query(nql)
    if resp.rows is not None:
        for row in resp.rows:
            path = row.columns[0].get_str()
            path_nodes = map(lambda x: long(x if x != '-1' else default_node), path.split('#'))
            path_cache[path_nodes[0]] = path_nodes
    for node in nodes:
        paths.append(path_cache[node])
    return np.asarray(paths, np.int64)
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
import time

import tensorflow as tf
import numpy as np
import random

from tf_euler.python.euler_ops import base
from tf_euler.python.euler_ops import type_ops

_sample_graph_label = base._LIB_OP.sample_graph_label

def _iter_body(i, state):
    y, count, n, ta = state
    curr_result = sample_node(count[i] * n, y[i])
    curr_result = tf.reshape(curr_result, [-1, n])
    out_ta = ta.write(i, curr_result)
    return i+1, (y, count, n, out_ta)

def sample_graph_label(count):
    if base.nebula_ops['sample_graph_label']:
        return nebula_sample_graph_label(count)
    return _sample_graph_label(count)
 
 
def nebula_sample_graph_label(count):
    labels = tf.py_func(
        _nebula_sample_graph_label,
        [count],
        tf.string,
        True,
        "NebulaSampleGraphLabel"
    )
    return tf.strings.regex_replace(labels, '\x00', '', True)
 
 
def _nebula_sample_graph_label(count):
    labels = []
    for i in range(count):
        labels.append(str(random.randint(base.nebula_graph_label_start, base.nebula_graph_label_end)))
    print(
        'sample_graph_label count[{}] start[{}] end[{}]'.format(
            count,
            base.nebula_graph_label_start,
            base.nebula_graph_label_end
        ))
    return np.asarray(labels, np.str)
 

def sample_node(count, node_type, condition=''):
    """
    Sample Nodes by specific types

    Args:
    count: A scalar tensor specify sample count
    types: A scalar tensor specify sample type
    condition: A atrribute specify sample condition

    Return:
    A 1-d tensor of sample node ids
    """
    if base.nebula_ops['sample_node']:
        return nebula_sample_node(count, node_type, condition)
    if node_type == '-1':
        types = -1
    else:
        types = type_ops.get_node_type_id(node_type)
    return base._LIB_OP.sample_node(count, types, condition)

def nebula_sample_node(count, node_type, condition=''):
    return base._LIB_OP.nebula_sample_node(count, node_type, base.nebula_space, condition)

def sample_edge(count, edge_type=None):
    """
    Sample Edges by specific types

    Args:
    count: A scalar tensor specify sample count
    types: A scalar tensor specify sample type

    Return:
    A 2-d tensor of sample edge ids
    """
    if base.nebula_ops['sample_edge']:
        return nebula_sample_edge(count, edge_type)
    if edge_type == '-1':
        types = -1
    else:
        types = type_ops.get_edge_type_id(edge_type)
    return base._LIB_OP.sample_edge(count, types)

def nebula_sample_edge(count, edge_type):
    edges = tf.py_func(
        _nebula_sample_edge,
        [count, edge_type],
        tf.int64,
        True,
        "NebulaSampleEdge"
    )
    return edges
 
def _nebula_sample_edge(count, edge_type):
    start_time = int(time.time() * 1000)
    edges = []
    for i in range(count):
        edges.append([0, 0, -1])
    nql = 'USE {}; SAMPLE EDGE {} LIMIT {}'.format(
        base.nebula_space,
        ', '.join('e_{}'.format(str(edge_type)) for edge_type in base.nebula_all_edge_types)
        if (edge_type == '-1' or edge_type == -1) else 'e_{}'.format(edge_type),
        count
    )
    resp = base.nebula_client.execute_query(nql)
    if resp.rows is not None and len(resp.rows) > 0:
        for i in range(count):
            row = resp.rows[i % len(resp.rows)]
            edge_type = row.columns[0].get_str()
            src_id = row.columns[1].get_id()
            dst_id = row.columns[2].get_id()
            # e_train->train->idx
            if edge_type.startswith('e_') and edge_type[2:] in base.nebula_all_edge_types:
                edge_type_idx = base.nebula_all_edge_types.index(edge_type[2:])
            else:
                edge_type_idx = -1
            edges[i] = [src_id, dst_id, edge_type_idx]
    resp_count = 0 if resp.rows is None else len(resp.rows)
    end_time = int(time.time() * 1000)
    print('sample_edge count[{}] edge_type[{}] start_at[{}] end_at[{}] duration[{}] resp_count[{}]'.format(
        count, edge_type, start_time, end_time, end_time - start_time, resp_count
    ))
    res = np.asarray(edges, np.int64)
    return res

def sample_node_with_src(src_nodes, count):
    """
    for each src node, sample "n" nodes with the same type

    Args:
      src_nodes: A 1-d `Tensor` of `int64`
      n: A scalar value of int
    Returns:
      A 2-dim Tensor, the first dim should be equal to dim of src_node.
      The second dim should be equal to n.
    """
    if base.nebula_ops['sample_node_with_src']:
        return nebula_sample_node_with_src(src_nodes, count)
    types = base._LIB_OP.get_node_type(src_nodes)
    return base._LIB_OP.sample_n_with_types(count, types)
def nebula_sample_node_with_src(src_nodes, count):
    nodes = tf.py_func(
        _nebula_sample_node_with_src,
        [src_nodes, count],
        tf.int64,
        True,
        "NebulaSampleNode"
    )
    return nodes
 
 
def _nebula_get_node_type(nodes):
    start_time = int(time.time() * 1000)
    nql = 'USE {}; '.format(base.nebula_space)
    nql += 'FETCH PROP ON * {} YIELD {}'.format(
        ','.join(str(x) for x in list(set(nodes))),
        ','.join('n_{}.w'.format(x) for x in base.nebula_all_node_types)
    )
    resp = base.nebula_client.execute_query(nql)
    node_type_cache = {}
    if resp.rows is not None:
        for row in resp.rows:
            node_type_idx = -1
            node_idx = row.columns[0].get_id()
            for ni in range(len(base.nebula_all_node_types)):
                weight = row.columns[ni + 1].get_double_precision()
                if weight > 0:
                    node_type_idx = ni
                    break
            if node_type_idx == -1:
                continue
            node_type_cache[node_idx] = base.nebula_all_node_types[node_type_idx]
    types = []
    for node in nodes:
        if node in node_type_cache:
            types.append(node_type_cache[node])
        else:
            types.append('-1')
    end_time = int(time.time() * 1000)
    print('get_node_type end_at[{}] duration[{}] in_count[{}] resp_count[{}]'.format(
            end_time,
            end_time - start_time,
            len(nodes),
            0 if (resp.rows is None) else len(resp.rows)
        ))
    return types
 
 
def _nebula_sample_node_with_src(src_nodes, count):
    sample_nodes = []
    start_time = int(time.time() * 1000)
    node_types = _nebula_get_node_type(src_nodes)
    for node_type in node_types:
        sample_nodes.append(_nebula_sample_node(count, node_type))
    end_time = int(time.time() * 1000)
    print('sample_node_with_src count[{}] end_at[{}] duration[{}] in_count[{}] resp_count[{}]'.format(
        count,
        end_time,
        end_time - start_time,
        len(src_nodes),
        len(sample_nodes)
    ))
    return np.asarray(sample_nodes, np.int64)
 


def get_graph_by_label(labels):
    res = base._LIB_OP.get_graph_by_label(labels)
    return tf.SparseTensor(*res[:3])

def nebula_get_graph_by_label(labels):
    indices, idx_values, shape = tf.py_func(
        _nebula_get_graph_by_label,
        [labels],
        [tf.int64, tf.int64, tf.int64],
        True,
        'NebulaGetGraphByLabel'
    )
    return tf.SparseTensor(indices, idx_values, shape)
 
def _nebula_get_graph_by_label(labels):
    start_time = int(time.time() * 1000)
    nql = 'USE {}; '.format(base.nebula_space)
    for label in labels:
        for node_type in base.nebula_all_node_types:
            nql += 'LOOKUP ON n_{} WHERE n_{}.b_graph_label == "{}" YIELD n_{}.b_graph_label as label UNION '.format(
                node_type, node_type, label, node_type
            )
    nql = nql[:-6]
    resp = base.nebula_client.execute_query(nql)
    idx_cache = {}
    if resp.rows is not None:
        for row in resp.rows:
            idx = row.columns[0].get_id()
            label = row.columns[1].get_str()
            if label not in idx_cache:
                idx_cache[label] = []
            idx_cache[label].append(idx)
    indices = []
    idx_values = []
    max_pos_x_count = 0
    for i in range(len(labels)):
        current_pos_x_count = 0
        label = labels[i]
        if label in idx_cache:
            for idx in idx_cache[label]:
                indices.append([i, current_pos_x_count])
                idx_values.append(idx)
                current_pos_x_count += 1
                if current_pos_x_count > max_pos_x_count:
                    max_pos_x_count = current_pos_x_count
    shape = [len(labels), max_pos_x_count]
    end_time = int(time.time() * 1000)
    print('get_graph_by_label labels[{}] end_at[{}] duration[{}] in_count[{}] resp_count[{}]'.format(
        ','.join(str(x) for x in labels),
        end_time,
        end_time - start_time,
        len(labels),
        0 if (resp.rows is None) else len(resp.rows)
    ))
    return np.asarray(indices, np.int64), np.asarray(idx_values, np.int64), np.asarray(shape, np.int64)


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

from collections import namedtuple

import tensorflow as tf

from tf_euler.python.euler_ops import base
from tf_euler.python.euler_ops import type_ops
from tf_euler.python.euler_ops import feature_ops
from tf_euler.python.utils.sample_util import *
import numpy as np
import time

_sample_neighbor_with_nebula = base._LIB_OP.nebula_sample_neighbor
_sample_neighbor = base._LIB_OP.sample_neighbor
_get_top_k_neighbor = base._LIB_OP.get_top_k_neighbor
_sample_fanout = base._LIB_OP.sample_fanout
_sample_neighbor_layerwise_with_adj = \
    base._LIB_OP.sample_neighbor_layerwise_with_adj
_sample_fanout_with_feature = base._LIB_OP.sample_fanout_with_feature


def _split_input_data(data_list, thread_num):
    size = tf.shape(data_list)[0]
    split_size = [size // thread_num] * (thread_num - 1)
    if thread_num == 1:
        split_size += [size]
    else:
        split_size += [-1]
    split_data_list = tf.split(data_list, split_size)
    return split_data_list


def sparse_get_adj(nodes, nb_nodes, edge_types, n=-1, m=-1):
    if base.nebula_ops['sparse_get_adj']:
        return nebula_sparse_get_adj(nodes, nb_nodes, edge_types, n, m)
    edge_types = type_ops.get_edge_type_id(edge_types)
    res = base._LIB_OP.sparse_get_adj(nodes, nb_nodes, edge_types, n, m)
    return tf.SparseTensor(*res[:3])

def nebula_sparse_get_adj(nodes, nb_nodes, edge_types, n=-1, m=-1):
    indices, values, shape = tf.py_func(
        _nebula_sparse_get_adj,
        [nodes, nb_nodes, edge_types, n, m],
        [tf.int64, tf.int64, tf.int64],
        True,
        'NebulaSparseGetAdj'
    )
    return tf.SparseTensor(indices, values, shape)
 
 
def _nebula_sparse_get_adj(nodes, nb_nodes, edge_types, n, m):
    start_time = int(time.time() * 1000)
    indices = []
    values = []
    shape = [1, len(nodes), len(nb_nodes)]
    src_idx_map = {}
    dst_idx_map = {}
    for i in range(len(nodes)):
        node = nodes[i]
        if node not in src_idx_map:
            src_idx_map[node] = []
        src_idx_map[node].append(i)
    for i in range(len(nb_nodes)):
        nb_node = nb_nodes[i]
        if nb_node not in dst_idx_map:
            dst_idx_map[nb_node] = []
        dst_idx_map[nb_node].append(i)
    nql = 'USE {}; GO {} STEPS FROM {} OVER {} YIELD {}, {}'.format(
        base.nebula_space,
        1,
        ', '.join(str(x) for x in nodes),
        ', '.join(('e_' + str(x)) for x in edge_types),
        ', '.join(('e_' + str(x) + '._src') for x in edge_types),
        ', '.join(('e_' + str(x) + '._dst') for x in edge_types)
    )
    nebula_start_time = int(time.time() * 1000)
    resp = base.nebula_client.execute_query(nql)
    nebula_cost_time = int(time.time() * 1000) - nebula_start_time
    if resp.rows is not None:
        for row in resp.rows:
            for i in range(len(edge_types)):
                src_id = row.columns[i].get_id()
                dst_id = row.columns[len(edge_types) + i].get_id()
                if src_id == 0 and dst_id == 0:
                    continue
                if dst_id not in dst_idx_map:
                    continue
                for src_idx in src_idx_map[src_id]:
                    for dst_idx in dst_idx_map[dst_id]:
                        indices.append([0, src_idx, dst_idx])
                        values.append(1)
                break
    end_time = int(time.time() * 1000)
    print('sparse_get_adj nodes[{}] nb_nodes[{}] edge_types[{}] end_at[{}] cost[{}] nebula_serv_cost[{}] nebula_resp_cost[{}] in_count[{}] resp_count[{}]'.format(
        ', '.join(str(x) for x in nodes),
        ', '.join(str(x) for x in nb_nodes),
        ', '.join(str(x) for x in edge_types),
        end_time,
        end_time - start_time,
        0 if (resp.rows is None) else int(resp.latency_in_us / 1000),
        nebula_cost_time,
        len(nodes),
        0 if (resp.rows is None) else len(resp.rows)
    ))
    return np.asarray(indices, np.int64), np.asarray(values, np.int64), np.asarray(shape, np.int64)
 


def sample_neighbor(nodes, edge_types, count, default_node=-1, condition=''):
    if base.nebula_ops['sample_neighbor']:
        split_data_list = _split_input_data(nodes, base.nebula_op_thread_num)
        split_result_list = [_sample_neighbor_with_nebula(split_data, edge_types, base.nebula_space,
                                                          base.nebula_all_edge_types, count, default_node)
                             for split_data in split_data_list]
        merge_neighbors = []
        merge_weights = []
        merge_types = []
        for i in range(len(split_result_list)):
            neighbors, weights, types = split_result_list[i]
            merge_neighbors.append(neighbors)
            merge_weights.append(weights)
            merge_types.append(types)
        return tf.concat(merge_neighbors, 0), tf.concat(merge_weights, 0), tf.concat(merge_types, 0)
    edge_types = type_ops.get_edge_type_id(edge_types)
    return _sample_neighbor(nodes, edge_types, count, default_node, condition)

def sample_neighbor_with_nebula(nodes, edge_types, count, default_node=-1, condition=''):
    result = _sample_neighbor_with_nebula(nodes, edge_types, count, default_node, condition)
    return result
 
 
def nebula_sample_neighbor(nodes, edge_types, count, default_node=-1, condition='', reuse=True):
    neighbors, weights, types = tf.py_func(
        _nebula_sample_neighbor_reuse if reuse else _nebula_sample_neighbor,
        [nodes, edge_types, count, default_node, condition],
        [tf.int64, tf.float32, tf.int32],
        True,
        "NebulaSampleNeighbor"
    )
    return neighbors, weights, types
 
def _nebula_sample_neighbor_reuse(nodes, edge_types, count, default_node, condition=''):
    start_time = int(time.time() * 1000)
    idx_dense = []
    weight_dense = []
    edge_dense = []
 
    unique_nodes = list(set(nodes))
    nql = 'USE {}; '.format(base.nebula_space)
    nql += 'sampleNB FROM {} OVER {} LIMIT {} '.format(
        ', '.join(str(x) for x in unique_nodes),
        ', '.join(('e_' + str(x)) for x in edge_types),
        count
    )
    nebula_start_time = int(time.time() * 1000)
    resp = base.nebula_client.execute_query(nql)
    nebula_cost_time = int(time.time() * 1000) - nebula_start_time
    idx_cache = {}
    weight_cache = {}
    edge_cache = {}
 
    if resp.rows is not None:
        for row in resp.rows:
            src_id = row.columns[2].get_integer()
            dst_id = row.columns[3].get_integer()
            weight = row.columns[4].get_double_precision()
            if src_id not in idx_cache:
                idx_cache[src_id] = []
                weight_cache[src_id] = []
                edge_cache[src_id] = []
            edge_type = row.columns[0].get_str()[2:]
            idx_cache[src_id].append(dst_id)
            weight_cache[src_id].append(weight)
            for e_id in range(len(base.nebula_all_edge_types)):
                if base.nebula_all_edge_types[e_id] == edge_type:
                    edge_cache[src_id].append(int(e_id))
                    break
    for ni in range(len(nodes)):
        node = nodes[ni]
        idx_dense.append([default_node] * count)
        weight_dense.append([0.0] * count)
        edge_dense.append([-1] * count)
        if node in idx_cache:
            for ci in range(count):
                idx_dense[ni][ci] = idx_cache[node][ci % len(idx_cache[node])]
                weight_dense[ni][ci] = weight_cache[node][ci % len(weight_cache[node])]
                edge_dense[ni][ci] = edge_cache[node][ci % len(edge_cache[node])]
 
    end_time = int(time.time() * 1000)
    print(
        'sample_neighbor end_at[{}] cost[{}] nebula_serv_cost[{}] nebula_resp_cost[{}] in_count[{}] resp_count[{}]'.format(
            end_time,
            end_time - start_time,
            0 if (resp.rows is None) else int(resp.latency_in_us / 1000),
            nebula_cost_time,
            len(nodes),
            0 if (resp.rows is None) else len(resp.rows)
        ))
    return np.asarray(idx_dense, np.int64), np.asarray(weight_dense, np.float32), np.asarray(edge_dense, np.int32)
 
def _nebula_sample_neighbor(nodes, edge_types, count, default_node, condition=''):
    start_time = int(time.time() * 1000)
    idx_dense = []
    weight_dense = []
    edge_dense = []
 
    nql = 'USE {}; '.format(base.nebula_space)
    nql += 'sampleNB FROM {} OVER {} LIMIT {} '.format(
        ', '.join(str(x) for x in nodes),
        ', '.join(('e_' + str(x)) for x in edge_types),
        count
    )
    nebula_start_time = int(time.time() * 1000)
    resp = base.nebula_client.execute_query(nql)
    nebula_cost_time = int(time.time() * 1000) - nebula_start_time
    idx_cache = {}
    weight_cache = {}
    edge_cache = {}
 
    if resp.rows is not None:
        for row in resp.rows:
            src_id = row.columns[2].get_integer()
            dst_id = row.columns[3].get_integer()
            weight = row.columns[4].get_double_precision()
            if src_id not in idx_cache:
                idx_cache[src_id] = [[]]
                weight_cache[src_id] = [[]]
                edge_cache[src_id] = [[]]
            if len(idx_cache[src_id][-1]) == count:
                idx_cache[src_id].append([])
            if len(weight_cache[src_id][-1]) == count:
                weight_cache[src_id].append([])
            if len(edge_cache[src_id][-1]) == count:
                edge_cache[src_id].append([])
            edge_type = row.columns[0].get_str()[2:]
            idx_cache[src_id][-1].append(dst_id)
            weight_cache[src_id][-1].append(weight)
            for e_id in range(len(base.nebula_all_edge_types)):
                if base.nebula_all_edge_types[e_id] == edge_type:
                    edge_cache[src_id][-1].append(int(e_id))
                    break
    pre_index_map = {}
    for ni in range(len(nodes)):
        node = nodes[ni]
        idx_dense.append([default_node] * count)
        weight_dense.append([0.0] * count)
        edge_dense.append([-1] * count)
        if node in idx_cache:
            if node not in pre_index_map:
                pre_index_map[node] = -1
            cur_index = (pre_index_map[node] + 1) % len(idx_cache[node])
            pre_index_map[node] = cur_index
            idx_data = idx_cache[node][cur_index]
            weight_data = weight_cache[node][cur_index]
            edge_data = edge_cache[node][cur_index]
            for ci in range(count):
                idx_dense[ni][ci] = idx_data[ci % len(idx_data)]
                weight_dense[ni][ci] = weight_data[ci % len(weight_data)]
                edge_dense[ni][ci] = edge_data[ci % len(edge_data)]
 
    end_time = int(time.time() * 1000)
    # print('sample_neighbor end_at[{}] cost[{}] nebula_serv_cost[{}] nebula_resp_cost[{}] in_count[{}] resp_count[{}]'.format(
    #     end_time,
    #     end_time - start_time,
    #     0 if (resp.rows is None) else int(resp.latency_in_us / 1000),
    #     nebula_cost_time,
    #     len(nodes),
    #     0 if (resp.rows is None) else len(resp.rows)
    # ))
    return np.asarray(idx_dense, np.int64), np.asarray(weight_dense, np.float32), np.asarray(edge_dense, np.int32)
 

def get_top_k_neighbor(nodes, edge_types, k, default_node=-1, condition=''):
    if base.nebula_ops['get_top_k_neighbor']:
        return nebula_get_top_k_neighbor(nodes, edge_types, k, default_node, condition)
    edge_types = type_ops.get_edge_type_id(edge_types)
    return _get_top_k_neighbor(nodes, edge_types, k, default_node, condition)

def nebula_get_top_k_neighbor(nodes, edge_types, k, default_node=-1, condition=''):
    sparse_ids, sparse_weights, sparse_types = nebula_get_full_neighbor(nodes, edge_types, 'weight', k, False, condition)
    GetTopKNeighbor = namedtuple('GetTopKNeighbor', ['neighbors', 'weights', 'types'])
    return GetTopKNeighbor(tf.sparse_tensor_to_dense(sparse_ids, default_value=default_node, validate_indices=False), \
        tf.sparse_tensor_to_dense(sparse_weights, validate_indices=False), \
        tf.sparse_tensor_to_dense(sparse_types, default_value=-1, validate_indices=False))
 


def sample_fanout_with_feature(nodes, edge_types, count, default_node,
                               dense_feature_names, dense_dimensions,
                               sparse_feature_names, sparse_default_values):
    if base.nebula_ops['sample_fanout_with_feature']:
        return nebula_sample_fanout_with_feature(nodes, edge_types, count, default_node,
            dense_feature_names, dense_dimensions, sparse_feature_names, sparse_default_values)
    edge_types = type_ops.get_edge_type_id(edge_types)
    res = _sample_fanout_with_feature(
        tf.reshape(nodes, [-1]), edge_types, count,
        default_node=default_node,
        sparse_feature_names=sparse_feature_names,
        sparse_default_values=sparse_default_values,
        dense_feature_names=dense_feature_names,
        dense_dimensions=dense_dimensions,
        N=len(count),
        ND=(len(count) + 1) * len(dense_feature_names),
        NS=(len(count) + 1) * len(sparse_feature_names))
    neighbors = [tf.reshape(nodes, [-1])]
    neighbors.extend([tf.reshape(i, [-1]) for i in res[0]])
    weights = res[1]
    types = res[2]
    dense_features = res[3]
    sparse_features = [tf.SparseTensor(*sp) for sp in zip(*res[4:7])]
    return neighbors, weights, types, dense_features, sparse_features

def nebula_sample_fanout_with_feature(nodes, edge_types, counts, default_node,
                               dense_feature_names, dense_dimensions,
                               sparse_feature_names, sparse_default_values):
    neighbors_list = [tf.reshape(nodes, [-1])]
    weights_list = []
    type_list = []
    dense_feature_list = []
    sparse_feature_list = []
    origin_nodes_count = nodes.shape.dims[0].value
    param_shapes = [origin_nodes_count, -1]
    for i in range(len(counts)):
        neighbors, weights, types = nebula_sample_neighbor(
            neighbors_list[-1],
            edge_types[i],
            counts[i],
            default_node
        )
        neighbors_list.append(tf.reshape(neighbors, [-1]))
        weights_list.append(tf.reshape(weights, param_shapes))
        type_list.append(tf.reshape(types, param_shapes))
        param_shapes.insert(-1, counts[i])
    for i in range(len(counts) + 1):
        dense_feature_list.extend(feature_ops.nebula_get_dense_feature(neighbors_list[i], dense_feature_names, dense_dimensions))
        sparse_feature_list.extend(feature_ops.nebula_get_sparse_feature(neighbors_list[i], sparse_feature_names, sparse_default_values))
    return neighbors_list, weights_list, type_list, dense_feature_list, sparse_feature_list
 


def sample_neighbor_layerwise(nodes, edge_types, count,
                              default_node=-1, weight_func=''):
    if base.nebula_ops['sample_neighbor_layerwise']:
        return nebula_sample_neighbor_layerwise(nodes, edge_types, count, default_node, weight_func)
    edge_types = type_ops.get_edge_type_id(edge_types)
    res = _sample_neighbor_layerwise_with_adj(nodes, edge_types, count,
                                              weight_func, default_node)
    return res[0], tf.SparseTensor(*res[1:4])

def nebula_sample_neighbor_layerwise(nodes, edge_types, count,
                                     default_node=-1, weight_func=''):
    neighbors, indices, values, shape = tf.py_func(
        _nebula_sample_neighbor_layerwise,
        [nodes, edge_types, count, default_node, weight_func],
        [tf.int64, tf.int64, tf.int64, tf.int64],
        True,
        'NebulaSampleNeighborLayerwise'
    )
    adj = tf.SparseTensor(indices, values, shape)
    return neighbors, adj
 
def _nebula_sample_neighbor_layerwise(nodes, edge_types, count,
                                      default_node=-1, weight_func=''):
    start_time = int(time.time() * 1000)
    batch = nodes.shape[0]
    n = nodes.shape[1]
    node_list = nodes.flatten()  # batch * n
    indices, idx_values, weight_values, _1, _2 = _nebula_get_full_neighbor(node_list, edge_types)
    src_idx_list = np.asarray(indices, np.int32)[..., 0]
    dst_list = idx_values
    w_list = weight_values.flatten()
    edge_set = set()  # stores all edges between neighbors
    node_w_list = []  # each node's sum weight(batch * n)
    w_index = -1  # last node's index
    for i in range(len(src_idx_list)):
        src_idx = src_idx_list[i]
        dst = dst_list[i]
        w = w_list[i]
        edge_set.add('{}->{}'.format(node_list[src_idx], dst))
        while w_index < src_idx - 1:  # encounter nodes with no neighbors
            node_w_list.append(0)
            w_index += 1
        if w_index != src_idx:  # w_index equals (src_idx - 1)
            node_w_list.append(w)
            w_index += 1
        else:
            node_w_list[w_index] += w
 
    # alias sample
    sample_node_list = []  # batch * count
    for bi in range(batch):
        fwc = FastWeightedCollection(node_list[bi * n : (bi + 1) * n], node_w_list[bi * n : (bi + 1) * n])
        if fwc.get_sum_weight == 0:
            for ci in range(count):
                sample_node_list.append(default_node)
        else:
            for ci in range(count):
                sample_node_list.append(fwc.sample()[0])
    neighbor_list, _, edge_list = _nebula_sample_neighbor(sample_node_list, edge_types, 1, default_node) # batch * count
    neighbor_list = neighbor_list.flatten()
 
    indices = []
    values = []
    shape = [batch, n, count]
    for bi in range(batch):
        for ni in range(n):
            src = nodes[bi][ni]
            for ci in range(count):
                dst = neighbor_list[bi * count + ci]
                edge = '{}->{}'.format(src, dst)
                if edge in edge_set:
                    indices.append([bi, ni, ci])
                    values.append(1)
                elif (ni == n - 1) and (ci == count - 1):
                    indices.append([bi, ni, ci])
                    values.append(0)
    end_time = int(time.time() * 1000)
    # print('sample_neighbor_layerwise nodes[{}] edge_types[{}] count[{}] end_at[{}] cost[{}] in_count[{}]'.format(
    #         ', '.join(str(x) for x in node_list),
    #         ', '.join(str(x) for x in edge_types),
    #         count,
    #         end_time,
    #         end_time - start_time,
    #         len(node_list)
    #     ))
    return np.reshape(neighbor_list, (batch, count)), indices, values, shape

def get_full_neighbor(nodes, edge_types, condition=''):
    """
    Args:
      nodes: A `Tensor` of `int64`.
      edge_types: A 1-D `Tensor` of int32. Specify edge types to filter
        outgoing edges.

    Return:
      A tuple of `SparseTensor` (neibors, weights).
        neighbors: A `SparseTensor` of `int64`.
        weights: A `SparseTensor` of `float`.
        types: A `SparseTensor` of `int32`
    """
    if base.nebula_ops['get_full_neighbor']:
        return nebula_get_full_neighbor(nodes, edge_types, '', 0, True, condition)
    edge_types = type_ops.get_edge_type_id(edge_types)
    sp_returns = base._LIB_OP.get_full_neighbor(nodes, edge_types, condition)
    return tf.SparseTensor(*sp_returns[:3]), \
        tf.SparseTensor(*sp_returns[3:6]), \
        tf.SparseTensor(*sp_returns[6:])

def nebula_get_full_neighbor(nodes, edge_types, sort='', limit=0, is_cut_tail=True, condition=''):
    indices, idx_values, weight_values, edge_values, shape = tf.py_func(
        _nebula_get_full_neighbor,
        [nodes, edge_types, sort, limit, is_cut_tail, condition],
        [tf.int64, tf.int64, tf.float32, tf.int32, tf.int64],
        True,
        'NebulaGetFullNeighbor_{}_{}'.format(sort, limit)
    )
    neighbors = tf.SparseTensor(indices, idx_values, shape)
    weights = tf.SparseTensor(indices, weight_values, shape)
    types = tf.SparseTensor(indices, edge_values, shape)
    return neighbors, weights, types
 
 
def _nebula_get_full_neighbor(nodes, edge_types, sort='', limit=0, is_cut_tail=True, condition=''):
    start_time = int(time.time() * 1000)
    nql = 'USE {}; GO {} STEPS FROM {} OVER {} YIELD {}, {}, {}'.format(
        base.nebula_space,
        1,
        ', '.join(str(x) for x in nodes),
        ', '.join(('e_' + str(x)) for x in edge_types),
        ', '.join(('e_' + str(x) + '._src') for x in edge_types),
        ', '.join(('e_' + str(x) + '._dst') for x in edge_types),
        ', '.join(('$$.n_' + str(x) + '.w') for x in base.nebula_all_node_types)
    )
    nebula_start_time = int(time.time() * 1000)
    resp = base.nebula_client.execute_query(nql)
    nebula_cost_time = int(time.time() * 1000) - nebula_start_time
    indices = []
    idx_values = []
    edge_values = []
    weight_values = []
    res_cache = {}
    if resp.rows is not None:
        for row in resp.rows:
            for i in range(len(edge_types)):
                src_id = row.columns[i].get_id()
                dst_id = row.columns[len(edge_types) + i].get_id()
                if src_id == 0 and dst_id == 0:
                    continue
                edge_type = resp.column_names[i][2:-5]
                edge_type_id = -1
                for e_id in range(len(base.nebula_all_edge_types)):
                    if base.nebula_all_edge_types[e_id] == edge_type:
                        edge_type_id = int(e_id)
                        break
                weight = 0.0
                for n in range(len(base.nebula_all_node_types)):
                    weight = row.columns[len(edge_types) * 2 + n].get_double_precision()
                    if weight > 0:
                        break
                if src_id not in res_cache:
                    res_cache[src_id] = []
                res_cache[src_id].append({
                    'dst_id': dst_id,
                    'edge_type_id': edge_type_id,
                    'weight': weight
                })
                break
    for src_id in res_cache:
        if sort == 'id':
            res_cache[src_id] = sorted(res_cache[src_id], key=lambda x: x['dst_id'])
        if sort == 'weight':
            res_cache[src_id] = sorted(res_cache[src_id], key=lambda x: x['weight'], reverse=True)
        if limit > 0:
            res_cache[src_id] = res_cache[src_id][:limit]
    max_res_count = 0
    tail_none_count = 0
    for i in range(len(nodes)):
        node = nodes[i]
        if node not in res_cache:
            tail_none_count += 1
            continue
        tail_none_count = 0
        res_list = res_cache[node]
        res_count = len(res_list)
        if res_count > max_res_count:
            max_res_count = res_count
        for j in range(res_count):
            res = res_list[j]
            indices.append([i, j])
            idx_values.append(res['dst_id'])
            edge_values.append(res['edge_type_id'])
            weight_values.append(res['weight'])
    shape = [len(nodes) - (tail_none_count if is_cut_tail else 0), max_res_count]
    end_time = int(time.time() * 1000)
    # print('get_full_neighbor sort[{}] limit[{}] end_at[{}] cost[{}] nebula_serv_cost[{}] nebula_resp_cost[{}] in_count[{}] resp_count[{}]'.format(
    #     sort,
    #     limit,
    #     end_time,
    #     end_time - start_time,
    #     0 if (resp.rows is None) else int(resp.latency_in_us / 1000),
    #     nebula_cost_time,
    #     len(nodes),
    #     0 if (resp.rows is None) else len(resp.rows)
    # ))
    return indices, idx_values, np.asarray(weight_values, np.float32), np.asarray(edge_values, np.int32), shape


def get_sorted_full_neighbor(nodes, edge_types, condition=''):
    """
    Args:
      nodes: A `Tensor` of `int64`.
      edge_types: A 1-D `Tensor` of int32. Specify edge types to filter
        outgoing edges.

    Return:
      A tuple of `SparseTensor` (neibors, weights).
        neighbors: A `SparseTensor` of `int64`.
        weights: A `SparseTensor` of `float`.
        types: A `SparseTensor` of `int32`
    """
    if base.nebula_ops['get_sorted_full_neighbor']:
        return nebula_get_sorted_full_neighbor(nodes, edge_types, condition)
    edge_types = type_ops.get_edge_type_id(edge_types)
    sp_returns = base._LIB_OP.get_sorted_full_neighbor(nodes,
                                                       edge_types,
                                                       condition)
    return tf.SparseTensor(*sp_returns[:3]),\
        tf.SparseTensor(*sp_returns[3:6]),\
        tf.SparseTensor(*sp_returns[6:])

def nebula_get_sorted_full_neighbor(nodes, edge_types, condition=''):
    return nebula_get_full_neighbor(nodes, edge_types, 'id', 0, True, condition)


def sample_fanout(nodes, edge_types, counts, default_node=-1):
    """
    Sample multi-hop neighbors of nodes according to weight in graph.

    Args:
      nodes: A 1-D `Tensor` of `int64`.
      edge_types: A list of 1-D `Tensor` of int32. Specify edge types to filter
        outgoing edges in each hop.
      counts: A list of `int`. Specify the number of sampling for each node in
        each hop.
      default_node: A `int`. Specify the node id to fill when there is no
        neighbor for specific nodes.

    Return:
      A tuple of list: (samples, weights)
        samples: A list of `Tensor` of `int64`, with the same length as
          `edge_types` and `counts`, with shapes `[num_nodes]`,
          `[num_nodes * count1]`, `[num_nodes * count1 * count2]`, ...
        weights: A list of `Tensor` of `float`, with shapes
          `[num_nodes * count1]`, `[num_nodes * count1 * count2]` ...
        types: A list of `Tensor` of `int32`, with shapes
          `[num_nodes * count1]`, `[num_nodes * count1 * count2]` ...
    """
    if base.nebula_ops['sample_fanout']:
        return nebula_sample_fanout(nodes, edge_types, counts, default_node)
    edge_types = [type_ops.get_edge_type_id(edge_type)
                  for edge_type in edge_types]
    neighbors_list = [tf.reshape(nodes, [-1])]
    weights_list = []
    type_list = []
    neighbors, weights, types = _sample_fanout(
        neighbors_list[-1],
        edge_types, counts,
        default_node=default_node,
        N=len(counts))
    neighbors_list.extend([tf.reshape(n, [-1]) for n in neighbors])
    weights_list.extend([tf.reshape(w, [-1]) for w in weights])
    type_list.extend([tf.reshape(t, [-1]) for t in types])
    return neighbors_list, weights_list, type_list

def nebula_sample_fanout(nodes, edge_types, counts, default_node=-1):
    neighbors_list = [tf.reshape(nodes, [-1])]
    weights_list = []
    type_list = []
    for i in range(len(counts)):
        neighbors, weights, types = nebula_sample_neighbor(
            neighbors_list[-1],
            edge_types[i],
            counts[i],
            default_node
        )
        neighbors_list.append(tf.reshape(neighbors, [-1]))
        weights_list.append(tf.reshape(weights, [-1]))
        type_list.append(tf.reshape(types, [-1]))
    return neighbors_list, weights_list, type_list
 

def sample_fanout_layerwise_each_node(nodes, edge_types, counts,
                                      default_node=-1):
    '''
      sample fanout layerwise for each node
    '''
    edge_types = [type_ops.get_edge_type_id(edge_type)
                  for edge_type in edge_types]
    neighbors_list = [tf.reshape(nodes, [-1])]
    adj_list = []
    for hop_edge_types, count in zip(edge_types, counts):
        if (len(neighbors_list) == 1):
            neighbors, _, _ = sample_neighbor(neighbors_list[-1],
                                              hop_edge_types,
                                              count,
                                              default_node=default_node)
            neighbors_list.append(tf.reshape(neighbors, [-1]))
        else:
            neighbors, adj = sample_neighbor_layerwise(
                tf.reshape(neighbors_list[-1], [-1, last_count]),
                hop_edge_types,
                count,
                default_node=default_node)
            neighbors_list.append(tf.reshape(neighbors, [-1]))
            adj_list.append(adj)
        last_count = count
    return neighbors_list, adj_list


def sample_fanout_layerwise(nodes, edge_types, counts,
                            default_node=-1, weight_func=''):
    edge_types = [type_ops.get_edge_type_id(edge_type)
                  for edge_type in edge_types]
    neighbors_list = [tf.reshape(nodes, [-1])]
    adj_list = []
    last_count = tf.size(nodes)
    for hop_edge_types, count in zip(edge_types, counts):
        neighbors, adj = sample_neighbor_layerwise(
            tf.reshape(neighbors_list[-1], [-1, last_count]),
            hop_edge_types,
            count,
            default_node=default_node,
            weight_func=weight_func)
        neighbors_list.append(tf.reshape(neighbors, [-1]))
        adj_list.append(adj)
        last_count = count
    return neighbors_list, adj_list


def get_multi_hop_neighbor(nodes, edge_types):
    """
    Get multi-hop neighbors with adjacent matrix.

    Args:
      nodes: A 1-D `tf.Tensor` of `int64`.
      edge_types: A list of 1-D `tf.Tensor` of `int32`. Specify edge types to
        filter outgoing edges in each hop.

    Return:
      A tuple of list: (nodes, adjcents)
        nodes: A list of N + 1 `tf.Tensor` of `int64`, N is the number of
          hops. Specify node set of each hop, including the root.
        adjcents: A list of N `tf.SparseTensor` of `int64`. Specify adjacent
          matrix between hops.
    """
    edge_types = [type_ops.get_edge_type_id(edge_type)
                  for edge_type in edge_types]
    nodes = tf.reshape(nodes, [-1])
    nodes_list = [nodes]
    adj_list = []
    for hop_edge_types in edge_types:
        neighbor, weight, _ = get_full_neighbor(nodes, hop_edge_types)
        next_nodes, next_idx = tf.unique(neighbor.values, out_idx=tf.int64)
        next_indices = tf.stack([neighbor.indices[:, 0], next_idx], 1)
        next_values = weight.values
        next_shape = tf.stack([tf.size(nodes), tf.size(next_nodes)])
        next_shape = tf.cast(next_shape, tf.int64)
        next_adj = tf.SparseTensor(next_indices, next_values, next_shape)
        next_adj = tf.sparse_reorder(next_adj)
        nodes_list.append(next_nodes)
        adj_list.append(next_adj)
        nodes = next_nodes
    return nodes_list, adj_list

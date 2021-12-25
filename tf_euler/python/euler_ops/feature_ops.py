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


def _split_input_data(data_list, thread_num):
    size = tf.shape(data_list)[0]
    split_size = [size // thread_num] * (thread_num - 1)
    if thread_num == 1:
        split_size += [size]
    else:
        split_size += [-1]
    split_data_list = tf.split(data_list, split_size)
    return split_data_list


def _get_sparse_feature(nodes_or_edges, feature_names, op, thread_num,
                        default_values=None):
    feature_names = map(str, feature_names)
    if not hasattr(feature_names, '__len__'):
        feature_names = list(feature_names)

    if default_values is None:
        default_values = [0] * len(feature_names)

    split_data_list = _split_input_data(nodes_or_edges, thread_num)
    split_result_list = [op(split_data, feature_names, default_values,
                         len(feature_names)) for split_data in split_data_list]
    split_sp = []
    for i in range(len(split_result_list)):
        split_sp.append(
            [tf.SparseTensor(*sp) for sp in zip(*split_result_list[i])])
    split_sp_transpose = map(list, zip(*split_sp))
    return [tf.sparse_concat(axis=0, sp_inputs=sp, expand_nonconcat_dim=True)
            for sp in split_sp_transpose]


def get_sparse_feature(nodes, feature_names,
                       default_values=None, thread_num=1):
    """
    Fetch sparse features of nodes.

    Args:
      nodes: A 1-d `Tensor` of `int64`.
      feature_names: A list of `int`. Specify uint64 feature ids in graph to
        fetch features for nodes.
      default_values: A `int`. Specify value to fill when there is no specific
        features for specific nodes.

    Return:
      A list of `SparseTensor` with the same length as `feature_names`.
    """
    if base.nebula_ops['get_sparse_feature']:
        return nebula_get_sparse_feature(nodes, feature_names, default_values, thread_num)
    return _get_sparse_feature(nodes, feature_names,
                               base._LIB_OP.get_sparse_feature, thread_num)

def nebula_get_sparse_feature(nodes, feature_names,
                              default_values=None, thread_num=1):
    sp_facts = tf.py_func(
        _nebula_get_sparse_feature,
        [nodes, feature_names, thread_num],
        [tf.int64, tf.int64, tf.int64] * len(feature_names),
        True,
        'NebulaGetSparseFeature'
    )
    sp_features = []
    for fi in range(len(feature_names)):
        sp_features.append(tf.SparseTensor(sp_facts[fi * 3], sp_facts[fi * 3 + 1], sp_facts[fi * 3 + 2]))
    return sp_features
 
 
def _nebula_get_sparse_feature(nodes, feature_names, thread_num):
    start_time = int(time.time() * 1000)
    sp_features = []
    nql = 'USE {}; FETCH PROP ON * {} YIELD {}, {}'.format(
        base.nebula_space,
        ', '.join(str(x) for x in nodes),
        ', '.join('n_{}.w'.format(n) for n in base.nebula_all_node_types),
        ', '.join('n_{}.s_{}'.format(n, f) for f in feature_names for n in base.nebula_all_node_types),
    )
    feature_cache = {}
    dim_cache = {}
    resp = base.nebula_client.execute_query(nql)
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
            if node_idx not in feature_cache:
                feature_cache[node_idx] = {}
            for fi in range(len(feature_names)):
                if fi not in dim_cache:
                    dim_cache[fi] = 0
                feature = row.columns[(fi + 1) * len(base.nebula_all_node_types) + 1 + node_type_idx].get_str()
                if feature is not None and len(feature) > 0:
                    feature_cache[node_idx][fi] = list(map(long, feature.split()))
                    this_dim = len(feature_cache[node_idx][fi])
                    if this_dim > dim_cache[fi]:
                        dim_cache[fi] = this_dim
    for fi in range(len(feature_names)):
        indices = []
        values = []
        shape = [len(nodes), dim_cache[fi]]
        for ni in range(len(nodes)):
            node = nodes[ni]
            if node in feature_cache and fi in feature_cache[node]:
                for vi in range(len(feature_cache[node][fi])):
                    indices.append([ni, vi])
                    values.append(feature_cache[node][fi][vi])
        sp_features.append(np.asarray(indices, np.int64))
        sp_features.append(np.asarray(values, np.int64))
        sp_features.append(np.asarray(shape, np.int64))
    end_time = int(time.time() * 1000)
    print('sparse_feature nodes[{}] feature_names[{}] thread_num[{}] end_at[{}] duration[{}] in_count[{}] resp_count[{}]'.format(
        ','.join(str(x) for x in nodes),
        ','.join(str(x) for x in feature_names),
        thread_num,
        end_time,
        end_time - start_time,
        len(nodes),
        0 if (resp.rows is None) else len(resp.rows)
    ))
    return sp_features
 

def get_edge_sparse_feature(edges, feature_names,
                            default_values=None, thread_num=1):
    """
    Args:
      edges: A 2-D `Tensor` of `int64`, with shape `[num_edges, 3]`.
      feature_names: A list of `int`. Specify uint64 feature ids in graph to
        fetch features for edges.
      default_values: A `int`. Specify value to fill when there is no specific
        features for specific edges.

    Return:
      A list of `SparseTensor` with the same length as `feature_names`.
    """
    if base.nebula_ops['get_edge_sparse_feature']:
        return nebula_get_edge_sparse_feature(edges, feature_names, default_values, thread_num)
    return _get_sparse_feature(edges, feature_names,
                               base._LIB_OP.get_edge_sparse_feature,
                               thread_num)
def nebula_get_edge_sparse_feature(edges, feature_names,
                              default_values=None, thread_num=1):
    sp_facts = tf.py_func(
        _nebula_get_edge_sparse_feature,
        [edges, feature_names, thread_num],
        [tf.int64, tf.int64, tf.int64] * len(feature_names),
        True,
        'NebulaGetEdgeSparseFeature'
    )
    sp_features = []
    for fi in range(len(feature_names)):
        sp_features.append(tf.SparseTensor(sp_facts[fi * 3], sp_facts[fi * 3 + 1], sp_facts[fi * 3 + 2]))
    return sp_features
 
 
def _nebula_get_edge_sparse_feature(edges, feature_names, thread_num):
    start_time = int(time.time() * 1000)
    sp_features = []
    feature_cache = {}
    dim_cache = {}
    edge_idx_map = {}
    resp_count = 0
    for edge in edges:
        edge_idx = '{}->{}'.format(edge[0], edge[1])
        edge_type = 'e_{}'.format(str(base.nebula_all_edge_types[int(edge[2])]))
        if edge_type not in edge_idx_map:
            edge_idx_map[edge_type] = []
        edge_idx_map[edge_type].append(edge_idx)
    for edge_type, edge_idx_list in edge_idx_map.items():
        nql = 'USE {}; FETCH PROP ON {} {} YIELD {}, {}'.format(
            base.nebula_space,
            edge_type,
            ', '.join(str(edge_idx) for edge_idx in edge_idx_list),
            '{}.w'.format(edge_type),
            ', '.join('{}.s_{}'.format(edge_type, f) for f in feature_names)
        )
        resp = base.nebula_client.execute_query(nql)
        resp_count += 0 if (resp.rows is None) else len(resp.rows)
        if resp.rows is not None:
            for row in resp.rows:
                # src dst rank w f1 f2 ...
                src_id = row.columns[0].get_id()
                dst_id = row.columns[1].get_id()
                edge_idx = '{}->{}-{}'.format(src_id, dst_id, edge_type)
                weight = row.columns[3].get_double_precision()
                if weight <= 0:
                    continue
                if edge_idx not in feature_cache:
                    feature_cache[edge_idx] = {}
                for fi in range(len(feature_names)):
                    if fi not in dim_cache:
                        dim_cache[fi] = 0
                    feature = row.columns[fi + 4].get_str()
                    if feature is not None and len(feature) > 0:
                        feature_cache[edge_idx][fi] = list(map(long, feature.split()))
                        this_dim = len(feature_cache[edge_idx][fi])
                        if this_dim > dim_cache[fi]:
                            dim_cache[fi] = this_dim
    for fi in range(len(feature_names)):
        indices = []
        values = []
        shape = [len(edges), dim_cache[fi]]
        for ei in range(len(edges)):
            edge = edges[ei]
            edge_idx = '{}->{}-e_{}'.format(edge[0], edge[1], str(base.nebula_all_edge_types[int(edge[2])]))
            if edge_idx in feature_cache and fi in feature_cache[edge_idx]:
                for vi in range(len(feature_cache[edge_idx][fi])):
                    indices.append([ei, vi])
                    values.append(feature_cache[edge_idx][fi][vi])
        sp_features.append(np.asarray(indices, np.int64))
        sp_features.append(np.asarray(values, np.int64))
        sp_features.append(np.asarray(shape, np.int64))
    end_time = int(time.time() * 1000)
    print('edge_sparse_feature edges[{}] feature_names[{}] thread_num[{}] end_at[{}] duration[{}] in_count[{}] resp_count[{}]'.format(
        ','.join(str(x) for x in edges),
        ','.join(str(x) for x in feature_names),
        thread_num,
        end_time,
        end_time - start_time,
        len(edges),
        resp_count
    ))
    return sp_features


def _get_dense_feature(nodes_or_edges, feature_names, dimensions,
                       op, thread_num):
    feature_names = map(str, feature_names)
    if not hasattr(feature_names, '__len__'):
        feature_names = list(feature_names)

    split_data_list = _split_input_data(nodes_or_edges, thread_num)
    split_result_list = [op(split_data,
                            feature_names,
                            dimensions,
                            N=len(feature_names))
                         for split_data in split_data_list]
    split_result_list_transpose = map(list, zip(*split_result_list))
    return [tf.concat(split_dense, 0)
            for split_dense in split_result_list_transpose]

def get_dense_feature_with_nebula(nodes_or_edges, feature_names, dimensions,
                       op, thread_num):
    feature_names = map(str, feature_names)
    if not hasattr(feature_names, '__len__'):
        feature_names = list(feature_names)
    n = len(feature_names)
    split_data_list = _split_input_data(nodes_or_edges, thread_num)
    split_result_list = [op(split_data,
                            feature_names,
                            dimensions,
                            n,
                            base.nebula_space,
                            base.nebula_all_node_types)
                         for split_data in split_data_list]
    split_result_list_transpose = map(list, zip(*split_result_list))
    return [tf.concat(split_dense, 0)
            for split_dense in split_result_list_transpose]
 

def get_dense_feature(nodes, feature_names, dimensions, thread_num=20):
    """
    Fetch dense features of nodes.

    Args:
      nodes: A 1-d `Tensor` of `int64`.
      feature_names: A list of `int`. Specify float feature ids in graph to
        fetch features for nodes.
      dimensions: A list of `int`. Specify dimensions of each feature.

    Return:
      A list of `Tensor` with the same length as `feature_names`.
    """
    if base.nebula_ops['get_dense_feature']:
        return get_dense_feature_with_nebula(nodes, feature_names, dimensions,
                           base._LIB_OP.nebula_get_dense_feature, base.nebula_op_thread_num)
    return _get_dense_feature(nodes, feature_names, dimensions,
                              base._LIB_OP.get_dense_feature, thread_num)

def nebula_get_dense_feature(nodes, feature_names, dimensions, thread_num=1):
    features = tf.py_func(
        _nebula_get_dense_feature,
        [nodes, feature_names, dimensions, thread_num],
        [tf.float32] * len(feature_names),
        True,
        "NebulaGetDenseFeature"
    )
    for fi in range(len(features)):
        if nodes.shape.dims is not None:
            features[fi].set_shape((nodes.shape.dims[0].value, dimensions[fi]))
    return features
 
def _nebula_get_dense_feature(nodes, feature_names, dimensions, thread_num):
    start_time = int(time.time() * 1000)
    features = []
    for fi in range(len(feature_names)):
        features.append([])
    nql = 'USE {}; FETCH PROP ON * {} YIELD {}, {}'.format(
        base.nebula_space,
        ', '.join(str(x) for x in nodes),
        ', '.join('n_{}.w'.format(n) for n in base.nebula_all_node_types),
        ', '.join('n_{}.d_{}'.format(n, f) for f in feature_names for n in base.nebula_all_node_types),
    )
    feature_cache = {}
    resp = base.nebula_client.execute_query(nql)
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
            if node_idx not in feature_cache:
                feature_cache[node_idx] = {}
            for fi in range(len(feature_names)):
                feature = row.columns[(fi + 1) * len(base.nebula_all_node_types) + 1 + node_type_idx].get_str()
                feature_dim = dimensions[fi]
                if feature is not None and len(feature) > 0:
                    feature_cache[node_idx][fi] = np.asarray(list(map(float, feature.split()))[:feature_dim], np.float32)
    for node in nodes:
        for fi in range(len(feature_names)):
            feature_dim = dimensions[fi]
            if node in feature_cache and fi in feature_cache[node]:
                features[fi].append(feature_cache[node][fi])
            else:
                features[fi].append(np.asarray([0.0] * feature_dim, np.float32))
    end_time = int(time.time() * 1000)
    print('dense_feature feature_names[{}] dimensions[{}] thread_num[{}] end_at[{}] duration[{}] in_count[{}] resp_count[{}]'.format(
        ','.join(str(x) for x in feature_names),
        ','.join(str(x) for x in dimensions),
        thread_num,
        end_time,
        end_time - start_time,
        len(nodes),
        0 if (resp.rows is None) else len(resp.rows)
    ))
    return features


def get_edge_dense_feature(edges, feature_names, dimensions, thread_num=1):
    """
    Fetch dense features of edges.

    Args:
      nodes: A 2-d `Tensor` of `int64`, with shape `[num_edges, 3]`.
      feature_names: A list of `int`. Specify float feature ids in graph to
        fetch features for edges.
      dimensions: A list of `int`. Specify dimensions of each feature.

    Return:
      A list of `Tensor` with the same length as `feature_names`.
    """
    if base.nebula_ops['get_edge_dense_feature']:
        return nebula_get_edge_dense_feature(edges, feature_names, dimensions, thread_num)
    return _get_dense_feature(edges, feature_names, dimensions,
                              base._LIB_OP.get_edge_dense_feature, thread_num)

def nebula_get_edge_dense_feature(edges, feature_names, dimensions, thread_num=1):
    features = tf.py_func(
        _nebula_get_edge_dense_feature,
        [edges, feature_names, dimensions, thread_num],
        [tf.float32] * len(feature_names),
        True,
        "NebulaGetEdgeDenseFeature"
    )
    for fi in range(len(features)):
        if edges.shape.dims is not None:
            features[fi].set_shape((edges.shape.dims[0].value, dimensions[fi]))
    return features
 
def _nebula_get_edge_dense_feature(edges, feature_names, dimensions, thread_num):
    start_time = int(time.time() * 1000)
    features = []
    resp_count = 0
    edge_idx_map = {}
    feature_cache = {}
    # test_case_file = 'testCase/' + str(start_time) + '.npy'
    # np.save(test_case_file, edges)
    for fi in range(len(feature_names)):
        features.append([])
    for edge in edges:
        edge_idx = '{}->{}'.format(edge[0], edge[1])
        edge_type = 'e_{}'.format(str(base.nebula_all_edge_types[int(edge[2])]))
        if edge_type not in edge_idx_map:
            edge_idx_map[edge_type] = []
        edge_idx_map[edge_type].append(edge_idx)
    for edge_type, edge_idx_list in edge_idx_map.items():
        nql = 'USE {}; FETCH PROP ON {} {} YIELD {}, {}'.format(
            base.nebula_space,
            edge_type,
            ', '.join(str(edge_idx) for edge_idx in edge_idx_list),
            '{}.w'.format(edge_type),
            ', '.join('{}.d_{}'.format(edge_type, f) for f in feature_names)
        )
        resp = base.nebula_client.execute_query(nql)
        resp_count += 0 if (resp.rows is None) else len(resp.rows)
        if resp.rows is not None:
            for row in resp.rows:
                # src dst rank w f1 f2 ...
                src_id = row.columns[0].get_id()
                dst_id = row.columns[1].get_id()
                edge_idx = '{}->{}'.format(src_id, dst_id)
                weight = row.columns[3].get_double_precision()
                if weight <= 0:
                    continue
                if edge_idx not in feature_cache:
                    feature_cache[edge_idx] = {}
                for fi in range(len(feature_names)):
                    feature = row.columns[fi + 4].get_str()
                    feature_dim = dimensions[fi]
                    if feature is not None and len(feature) > 0:
                        feature_cache[edge_idx][fi] = np.asarray(list(map(float, feature.split()))[:feature_dim],
                                                                 np.float32)
    for edge in edges:
        edge_idx = '{}->{}'.format(edge[0], edge[1])
        for fi in range(len(feature_names)):
            feature_dim = dimensions[fi]
            if edge_idx in feature_cache and fi in feature_cache[edge_idx]:
                features[fi].append(feature_cache[edge_idx][fi])
            else:
                features[fi].append(np.asarray([0.0] * feature_dim, np.float32))
                # info = 'edge[{}] resp_count[{}]\n'.format(
                #             str(edge),
                #             resp_count
                #         )
                # print(info)
                # with open ('not_found.txt', 'a') as ff:
                #     ff.write(info)
    end_time = int(time.time() * 1000)
    print('edge_dense_feature edges[{}] feature_names[{}] dimensions[{}] thread_num[{}] end_at[{}] duration[{}] in_count[{}] resp_count[{}]'.format(
        ','.join(str(x) for x in edges),
        ','.join(str(x) for x in feature_names),
        ','.join(str(x) for x in dimensions),
        thread_num,
        end_time,
        end_time - start_time,
        len(edges),
        resp_count
    ))
    return features

def _get_binary_feature(nodes_or_edges, feature_names, op, thread_num):
    feature_names = map(str, feature_names)
    if not hasattr(feature_names, '__len__'):
        feature_names = list(feature_names)

    split_data_list = _split_input_data(nodes_or_edges, thread_num)
    split_result_list = [op(split_data, feature_names, N=len(feature_names))
                         for split_data in split_data_list]
    split_result_list_transpose = map(list, zip(*split_result_list))
    return [tf.concat(split_binary, 0)
            for split_binary in split_result_list_transpose]


def get_binary_feature(nodes, feature_names, thread_num=1):
    """
    Fetch binary features of nodes.

    Args:
      nodes: A 1-d `Tensor` of `int64`.
      feature_names: A list of `int`. Specify uint64 feature ids in graph to
        fetch features for nodes.

    Return:
      A list of `String Tensor` with the same length as `feature_names`.
    """
    if base.nebula_ops['get_binary_feature']:
        return nebula_get_binary_feature(nodes, feature_names, thread_num)
    return _get_binary_feature(nodes, feature_names,
                               base._LIB_OP.get_binary_feature, thread_num)

def nebula_get_binary_feature(nodes, feature_names, thread_num=1):
    features = tf.py_func(
        _nebula_get_binary_feature,
        [nodes, feature_names, thread_num],
        [tf.string] * len(feature_names),
        True,
        "NebulaGetBinaryFeature"
    )
    for fi in range(len(features)):
        if nodes.shape.dims is not None:
            features[fi].set_shape((nodes.shape.dims[0].value,))
    return features
 
def _nebula_get_binary_feature(nodes, feature_names, thread_num):
    start_time = int(time.time() * 1000)
    features = []
    for fi in range(len(feature_names)):
        features.append([])
    nql = 'USE {}; FETCH PROP ON * {} YIELD {}, {}'.format(
        base.nebula_space,
        ', '.join(str(x) for x in nodes),
        ', '.join('n_{}.w'.format(n) for n in base.nebula_all_node_types),
        ', '.join('n_{}.b_{}'.format(n, f) for f in feature_names for n in base.nebula_all_node_types),
    )
    feature_cache = {}
    resp = base.nebula_client.execute_query(nql)
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
            if node_idx not in feature_cache:
                feature_cache[node_idx] = {}
            for fi in range(len(feature_names)):
                feature = row.columns[(fi + 1) * len(base.nebula_all_node_types) + 1 + node_type_idx].get_str()
                if feature is not None and len(feature) > 0:
                    feature_cache[node_idx][fi] = feature
    for node in nodes:
        for fi in range(len(feature_names)):
            if node in feature_cache and fi in feature_cache[node]:
                features[fi].append(feature_cache[node][fi])
            else:
                features[fi].append('')
    end_time = int(time.time() * 1000)
    print('binary_feature nodes[{}] feature_names[{}] thread_num[{}] end_at[{}] duration[{}] in_count[{}] resp_count[{}]'.format(
        ','.join(str(x) for x in nodes),
        ','.join(str(x) for x in feature_names),
        thread_num,
        end_time,
        end_time - start_time,
        len(nodes),
        0 if (resp.rows is None) else len(resp.rows)
    ))
    return features

def get_edge_binary_feature(edges, feature_names, thread_num=1):
    """
    Fetch binary features of edges.

    Args:
      edges: A 2-d `Tensor` of `int64`, with shape `[num_edges, 3]`.
      feature_names: A list of `int`. Specify uint64 feature ids in graph to
        fetch features for nodes.

    Return:
      A list of `String Tensor` with the same length as `feature_names`.
    """
    if base.nebula_ops['get_edge_binary_feature']:
        return nebula_get_edge_binary_feature(edges, feature_names, thread_num)
    return _get_binary_feature(edges, feature_names,
                               base._LIB_OP.get_edge_binary_feature,
                               thread_num)

def nebula_get_edge_binary_feature(edges, feature_names, thread_num=1):
    features = tf.py_func(
        _nebula_get_edge_binary_feature,
        [edges, feature_names, thread_num],
        [tf.string] * len(feature_names),
        True,
        "NebulaGetEdgeBinaryFeature"
    )
    for fi in range(len(features)):
        if edges.shape.dims is not None:
            features[fi].set_shape((edges.shape.dims[0].value,))
    return features
 
def _nebula_get_edge_binary_feature(edges, feature_names, thread_num):
    start_time = int(time.time() * 1000)
    features = []
    feature_cache = {}
    edge_idx_map = {}
    resp_count = 0
    for fi in range(len(feature_names)):
        features.append([])
    for edge in edges:
        edge_idx = '{}->{}'.format(edge[0], edge[1])
        edge_type = 'e_{}'.format(str(base.nebula_all_edge_types[int(edge[2])]))
        if edge_type not in edge_idx_map:
            edge_idx_map[edge_type] = []
        edge_idx_map[edge_type].append(edge_idx)
    for edge_type, edge_idx_list in edge_idx_map.items():
        nql = 'USE {}; FETCH PROP ON {} {} YIELD {}, {}'.format(
            base.nebula_space,
            edge_type,
            ', '.join(str(edge_idx) for edge_idx in edge_idx_list),
            '{}.w'.format(edge_type),
            ', '.join('{}.b_{}'.format(edge_type, f) for f in feature_names)
        )
        resp = base.nebula_client.execute_query(nql)
        resp_count += 0 if (resp.rows is None) else len(resp.rows)
        if resp.rows is not None:
            for row in resp.rows:
                # src dst rank w f1 f2 ...
                src_id = row.columns[0].get_id()
                dst_id = row.columns[1].get_id()
                edge_idx = '{}->{}-{}'.format(src_id, dst_id, edge_type)
                weight = row.columns[3].get_double_precision()
                if weight <= 0:
                    continue
                if edge_idx not in feature_cache:
                    feature_cache[edge_idx] = {}
                for fi in range(len(feature_names)):
                    feature = row.columns[fi + 4].get_str()
                    if feature is not None and len(feature) > 0:
                        feature_cache[edge_idx][fi] = feature
 
    for edge in edges:
        edge_idx = '{}->{}-e_{}'.format(edge[0], edge[1], str(base.nebula_all_edge_types[int(edge[2])]))
        for fi in range(len(feature_names)):
            if edge_idx in feature_cache and fi in feature_cache[edge_idx]:
                features[fi].append(feature_cache[edge_idx][fi])
            else:
                features[fi].append('')
    end_time = int(time.time() * 1000)
    print('edge_binary_feature edges[{}] feature_names[{}] thread_num[{}] end_at[{}] duration[{}] in_count[{}] resp_count[{}]'.format(
        ','.join(str(x) for x in edges),
        ','.join(str(x) for x in feature_names),
        thread_num,
        end_time,
        end_time - start_time,
        len(edges),
        resp_count
    ))
    return features
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import ctypes
import numpy as np

from tempfile import TemporaryDirectory
from lpot.utils.utility import LazyImport
from lpot.model.model import MXNetModel as LPOTModel

mx = LazyImport("mxnet")


QUANTIZE_OP_NAME = 'quantize_output'
QUANTIZE_DEFAULT_ALGORITHM = 'minmax'


def check_mx_version(version):
    """Checks MXNet version.

    Args:
        version (str): version to check.

    Returns:
        boolen: True if mx.__version__ >= version, else False.
    """
    d1 = re.split(r'\.', mx.__version__)
    d2 = re.split(r'\.', version)
    d1 = [int(d1[i]) for i in range(len(d1))]
    d2 = [int(d2[i]) for i in range(len(d2))]
    return d1 >= d2


def isiterable(obj) -> bool:
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def ensure_list(x):
    return x if isinstance(x, (tuple, list)) else [x]


def make_lpot_model(target, qsym_model, ctx, provide_data):
    assert isinstance(qsym_model, tuple) and isinstance(qsym_model[0], mx.symbol.Symbol)

    if isinstance(target.model, mx.gluon.HybridBlock):
        return LPOTModel(make_symbol_block(qsym_model, ctx, provide_data))
    return LPOTModel(qsym_model)


def fuse(sym_model, ctx):
    assert isinstance(sym_model, tuple) and isinstance(sym_model[0], mx.symbol.Symbol)

    symnet, args, auxs = sym_model
    if ctx == mx.cpu():
        symnet = symnet.get_backend_symbol('MKLDNN_QUANTIZE')
    return (symnet, args, auxs)


def prepare_model_data(lpot_model, ctx, dataloader):
    """Extracts symnet, args, auxs from model and wraps the dataloader in a DataLoaderWrap

    Args:
        lpot_model (object): Model to extract from.
        dataloader (object): Dataloader to wrap
    Returns:
        tuple: Symbol model (symnet, args, auxs) and DataLoaderWrap.
    """
    assert isinstance(lpot_model, LPOTModel)

    model = lpot_model.model
    if isinstance(model, mx.gluon.HybridBlock):
        sym_model, dataloader = _get_HybridBlock_symnet(model, ctx, dataloader)
    elif isinstance(model[0], mx.symbol.Symbol):
        sym_model, dataloader = _get_model_symnet(model, dataloader)
    else:
        raise TypeError('Wrong model type')
    return (fuse(sym_model, ctx), dataloader)


def _get_HybridBlock_symnet(network, ctx, dataloader):
    if isinstance(dataloader, mx.io.DataIter):
        dataloader = DataIterLoader(dataloader)
    assert isiterable(dataloader), 'Dataloader must be iterable (mx.gluon.data.DataLoader-like)'

    data = ensure_list(next(iter(dataloader)))
    data = [mx.nd.zeros(d.shape, dtype=d.dtype, ctx=ctx) for d in data]  # data example
    network.hybridize()
    while True:
        try:
            network(*data)
        except:
            del data[-1]  # remove label
        else:
            break

    inputs, _ = network._cached_graph
    provide_data = [mx.io.DataDesc(name=i.name, shape=d.shape) for i, d in zip(inputs, data)]
    with TemporaryDirectory() as tmpdirname:
        prefix = os.path.join(tmpdirname, 'tmp')
        network.export(prefix, epoch=0)
        model = mx.model.load_checkpoint(prefix, 0)

    if not isinstance(dataloader, DataLoaderWrap):
        dataloader = DataLoaderWrap(dataloader, provide_data)
    return (model, dataloader)


def _get_model_symnet(model, dataloader):
    assert hasattr(dataloader, 'provide_data'), \
        'Dataloader must provide data information (mx.data.DataDesc for each input)'
    provide_data = dataloader.provide_data

    if isinstance(dataloader, mx.io.DataIter):
        dataloader = DataIterLoader(dataloader)
    assert isiterable(dataloader), 'Dataloader must be iterable (mx.gluon.data.DataLoader-like)'

    if not isinstance(dataloader, DataLoaderWrap):
        dataloader = DataLoaderWrap(dataloader, provide_data)
    return (model, dataloader)


def is_model_quantized(model):
    assert isinstance(model, tuple) and isinstance(model[0], mx.symbol.Symbol)

    for sym in model[0].get_internals():
        _, was_quantized = _dequantize_op_sym_name(sym)
        if was_quantized:
            return True
    return False


def _dequantize_op_sym_name(sym, check_list=None):
    QUANTIZED_PREFIX = 'quantized_'
    name = sym.name
    if name.startswith(QUANTIZED_PREFIX):
        name = name[len(QUANTIZED_PREFIX):]
        assert check_list is None or name in check_list, \
            'name of the quantized symbol must be in the following format: ' \
            '({}_<fp32_sym_name>). Symbol: {}'.format(QUANTIZED_PREFIX, name)
    return (name, sym.name != name)


def query_quantizable_ops(sym_model, ctx, dataloader):
    """Query quantizable ops of the given model.

    Args:
        model (tuple): model to query.

    Returns:
        list: quantizable ops of the given model.
    """
    assert isinstance(sym_model, tuple) and isinstance(sym_model[0], mx.symbol.Symbol)
    assert isinstance(dataloader, DataLoaderWrap)

    symnet = sym_model[0]
    nodes_ops = {n['name']: n['op'] for n in json.loads(symnet.tojson())['nodes']}

    qmodel, calib_tensors = quantize_model(sym_model, {'ctx': ctx,
                                                       'quantized_dtype': 'auto',
                                                       'quantize_mode': 'smart'})
    qsymnet = qmodel[0]
    qnodes_ops = {n['name']: n['op'] for n in json.loads(qsymnet.tojson())['nodes']}

    QUANTIZE_OP_NAMES = ['_contrib_quantize_v2']

    # Getting quantizable nodes:
    # 1. Get nodes that has been quantized
    # 2. Get nodes whose outputs has been quantized (by 'quantize' nodes) - these are
    #    inputs of the nodes that has been quantized
    quantizable = {}
    for qsym in qsymnet.get_internals():
        sym_name, was_quantized = _dequantize_op_sym_name(qsym, nodes_ops.keys())
        if was_quantized:
            quantizable[sym_name] = nodes_ops[sym_name]
        elif qnodes_ops[qsym.name] in QUANTIZE_OP_NAMES:
            assert qsym.name not in nodes_ops.keys(), \
                'quantize nodes must not be present in the fp32 model'

            assert len(qsym.get_children()) == 1, \
                '`quantize` node should only have one input'

            q_in_sym = qsym.get_children()[0]
            assert q_in_sym.name in nodes_ops.keys(), \
                'name of the `quantize` input node must be the same' \
                'as the name of the corresponding node in the fp32 model'

            # quantize nodes do not exist in the fp32 model, so
            # we treat their input nodes as their fp32 equivalent
            quantizable[q_in_sym.name] = QUANTIZE_OP_NAME

    quantizable_ops = [{'name': name, 'type': op}
                       for (name, op) in quantizable.items()]

    # getting tensors to nodes mapping (for adaptor.inspect_tensor)

    def b_filter():
        yield True  # 1 batch

    # collect fp32 tensors
    collector = NameCollector()
    run_forward(sym_model, ctx, dataloader, b_filter(), collector)
    tensors = set(collector.names)

    # map tensors to nodes
    tensor_to_node = {}
    nodes = set(nodes_ops.keys())
    for tensor in tensors:
        node = _tensor_to_node(tensor, nodes)
        if node != '':
            tensor_to_node[tensor] = node

    assert set(calib_tensors).issubset(set(tensor_to_node.keys()))
    assert set(quantizable.keys()).issubset(set(tensor_to_node.values()))

    return quantizable_ops, tensor_to_node


def quantize_model(sym_model, qconfig):
    assert isinstance(sym_model, tuple) and isinstance(sym_model[0], mx.symbol.Symbol)

    symnet, args, auxs = sym_model
    qconfig['offline_params'] = list(args.keys())
    qsymnet, calib_tensors = mx.contrib.quantization._quantize_symbol(symnet, **qconfig)
    # args = mx.contrib.quantization._quantize_params(qsymnet, args, {})
    return ((qsymnet, args, auxs), calib_tensors)


def _tensor_to_node(tensor, nodes):
    """Map tensor to one of the nodes. This function assumes, that node tensors 
    (weights, outputs, etc) contain node name in their names
    """
    assert len(nodes) > 0, '`nodes` cannot be empty'

    PATTERNS = {'',
                '_output[0-9]*$',
                '_[0-9]+$'}
    mapping = []
    for pattern in PATTERNS:
        node = re.sub(pattern, '', tensor)
        if node in nodes and node not in mapping:
            mapping.append(node)
            assert len(mapping) == 1, 'Tensor matched to more than one node. ' \
                'Tensor: {}, matched: {}'.format(tensor, mapping)
    return mapping[0] if len(mapping) > 0 else ''


def _qtensor_to_tensor(qtensor, tensors):
    """Map quantized tensor to its fp32 equivalent. New ops may require updating the patterns.
    Tensors of quantize nodes (which are not present in fp32 models) will be mapped to their input
    nodes
    """
    assert len(tensors) > 0, '`tensors` cannot be empty'

    PATTERNS = {'_quantize_output0': '',
                '_quantize_0': '',
                '_0_quantize_output0': '_output',
                '_0_quantize_0': '_output',
                '_([0-9]+)_quantize_output0': '_output\g<1>',
                '_([0-9]+)_quantize_0': '_output\g<1>',
                'quantized_': ''}
    mapping = []
    for pattern, repl in PATTERNS.items():
        tensor = re.sub(pattern, repl, qtensor)
        if tensor in tensors and tensor not in mapping:
            mapping.append(tensor)
            assert len(mapping) == 1, \
                'Quantized tensor matched more than one fp32 tensor. ' \
                'Quantized tensor: {}, matched: {}'.format(qtensor, mapping)
    return mapping[0] if len(mapping) > 0 else ''


def run_forward(sym_model, ctx, dataloader, b_filter, collector=None, pre_batch=None, post_batch=None):
    assert isinstance(dataloader, DataLoaderWrap)
    assert collector is None or (hasattr(collector, 'collect_gluon') and
                                 hasattr(collector, 'collect_module'))

    if check_mx_version('2.0.0'):
        sym_block = make_symbol_block(sym_model, ctx, dataloader.provide_data)
        if collector is not None:
            sym_block.register_op_hook(collector.collect_gluon, monitor_all=True)
        return _gluon_forward(sym_block, ctx, dataloader, b_filter, pre_batch, post_batch)
    else:
        mod = make_module(sym_model, ctx, dataloader.provide_data)
        if collector is not None:
            mod._exec_group.execs[0].set_monitor_callback(
                collector.collect_module, monitor_all=True)
        return _module_forward(mod, dataloader, b_filter, pre_batch, post_batch)


def make_symbol_block(sym_model, ctx, provide_data):
    assert isinstance(sym_model, tuple) and isinstance(sym_model[0], mx.symbol.Symbol)

    symnet, args, auxs = sym_model
    inputs = [mx.sym.var(d.name) for d in provide_data]
    net = mx.gluon.SymbolBlock(symnet, inputs)
    param_dict = args
    param_dict.update(auxs)
    if check_mx_version('2.0.0'):
        net.load_dict(param_dict, cast_dtype=True, dtype_source='saved')
    else:
        # params = {'arg:' + name: param for name, param in args.items()}
        # params.update({'aux:' + name: param for name, param in auxs.items()})
        net.collect_params().load_dict(param_dict, ctx=ctx, cast_dtype=True, dtype_source='saved')
    net.hybridize(static_alloc=False, static_shape=False)
    return net


def make_module(sym_model, ctx, provide_data):
    assert isinstance(sym_model, tuple) and isinstance(sym_model[0], mx.symbol.Symbol)

    symnet, args, auxs = sym_model
    mod = mx.module.module.Module(symbol=symnet,
                                  data_names=[d.name for d in provide_data],
                                  label_names=None,
                                  context=ctx)
    mod.bind(provide_data, for_training=False)
    mod.set_params(args, auxs, allow_missing=True)
    return mod


def _gluon_forward(net, ctx, dataloader, b_filter, pre_batch=None, post_batch=None):
    batch_num = 0
    for run, batch in zip(b_filter, dataloader):
        if not run:
            continue
        batch_num += 1
        batch = [d.as_in_context(ctx) for d in ensure_list(batch)]
        data = batch[:len(dataloader.provide_data)]
        label = batch[len(dataloader.provide_data):]

        if pre_batch is not None:
            pre_batch(net, (data, label))

        out = net(*data)

        if post_batch is not None:
            post_batch(net, (data, label), out)
    return batch_num


def _module_forward(module, dataloader, b_filter, pre_batch=None, post_batch=None):
    batch_num = 0
    for run, batch in zip(b_filter, dataloader):
        if not run:
            continue
        batch_num += 1
        data = batch[:len(dataloader.provide_data)]
        label = batch[len(dataloader.provide_data):]

        if pre_batch is not None:
            pre_batch(module, (data, label))

        module.forward(mx.io.DataBatch(data=data), is_train=False)

        if post_batch is not None:
            post_batch(module, (data, label), module.get_outputs())
    return batch_num


def parse_tune_config(tune_cfg, quantizable_ops, ctx):
    """Convert the strategy config to MXNet quantization config.

    Args:
        tune_cfg (dict): tune config from lpot strategy.
    """
    excluded_symbols = []
    calib_minmax_nodes = set()
    calib_kl_nodes = set()
    quantize_node_cfg = None
    for (node_name, op), d in tune_cfg['op'].items():
        if op == QUANTIZE_OP_NAME:
            quantize_node_cfg = d['activation']['algorithm']
            break
    # assert quantize_node_cfg is not None, 'There must always be at least one quantize node'
    if quantize_node_cfg is None:
        quantize_node_cfg = QUANTIZE_DEFAULT_ALGORITHM

    for op in quantizable_ops:
        cfg = tune_cfg['op'][(op['name'], op['type'])]['activation']
        if cfg['dtype'] in ['fp32', 'bf16']:
            excluded_symbols.append(op['name'])
            # config for quantize node, that might be added in place of this node
            # (to quantize its output)
            cfg['algorithm'] = quantize_node_cfg
        if cfg['algorithm'] == 'kl':
            calib_kl_nodes.add(op['name'])
        else:
            calib_minmax_nodes.add(op['name'])
    assert len(calib_kl_nodes & calib_minmax_nodes) == 0

    quant_cfg = {'ctx': ctx,
                 'excluded_symbols': excluded_symbols,
                 'quantized_dtype': 'auto',
                 'quantize_mode': 'smart'}
    if check_mx_version('1.7.0'):
        quant_cfg['quantize_granularity'] = 'tensor-wise'

    calib_cfg = {'ctx': quant_cfg['ctx'],
                 'quantized_dtype': quant_cfg['quantized_dtype'],
                 'batches': tune_cfg['calib_iteration'],
                 'calib_mode': 'naive',
                 'calib_kl_nodes': calib_kl_nodes,
                 'calib_minmax_nodes': calib_minmax_nodes}

    return quant_cfg, calib_cfg


def distribute_calib_tensors(calib_tensors, calib_cfg, tensor_to_node):
    calib_tensors = set(calib_tensors)
    kl_tensors = {}
    minmax_tensors = {}
    for cl in calib_tensors:
        assert cl in tensor_to_node, '`calib_tensors` entry matched no node. Entry: {}'.format(cl)
        node = tensor_to_node[cl]
        if node in calib_cfg['calib_kl_nodes']:
            kl_tensors[cl] = node
        if node in calib_cfg['calib_minmax_nodes']:
            minmax_tensors[cl] = node

    kl_tensors = set(kl_tensors.keys())
    minmax_tensors = set(minmax_tensors.keys())
    assert len(kl_tensors & minmax_tensors) == 0, 'same `calib_tensors` entries matched both kl ' \
        'and minmax nodes. Entries: {}'.format(kl_tensors & minmax_tensors)

    rest = calib_tensors - (kl_tensors | minmax_tensors)
    assert len(rest) == 0, 'Unexpected `calib_tensors` entries. Entries: {}'.format(rest)

    return (kl_tensors, minmax_tensors)


def calib_model(qsym_model, collector, calib_cfg, logger=None):
    assert isinstance(qsym_model, tuple) and isinstance(qsym_model[0], mx.symbol.Symbol)

    qsymnet, qargs, auxs = qsym_model
    if check_mx_version('2.0.0'):
        return mx.contrib.quantization.calib_graph(
            qsymnet, qargs, auxs, collector, calib_cfg['calib_mode'], logger=logger)
    else:
        return mx.contrib.quantization.calib_graph(
            qsymnet, qargs, auxs, collector, calib_cfg['calib_mode'],
            quantized_dtype=calib_cfg['quantized_dtype'], logger=logger)


def _get_kl_thresholds(hist_dict, dtype, logger):
    if check_mx_version('2.0.0'):
        return mx.contrib.quantization._LayerHistogramCollector.get_optimal_thresholds(
            hist_dict, dtype, logger=logger)
    else:
        return mx.contrib.quantization._get_optimal_thresholds(hist_dict, dtype, logger=logger)


def _combine_histogram(old_hist, arr, new_min, new_max, new_th):
    if check_mx_version('2.0.0'):
        return mx.contrib.quantization._LayerHistogramCollector.combine_histogram(
            old_hist, arr, new_min, new_max, new_th)
    else:
        return mx.contrib.quantization.combine_histogram(old_hist, arr, new_min, new_max, new_th)


class DataLoaderWrap:
    def __init__(self, dataloader, provide_data):
        self.dataloader = dataloader
        self.provide_data = provide_data
        self._iter = None

    def __iter__(self):
        self._iter = iter(self.dataloader)
        return self

    def __next__(self):
        return next(self._iter)


class DataIterLoader:
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __iter__(self):
        self.data_iter.reset()
        return self

    def __next__(self):
        batch = self.data_iter.__next__()
        return batch.data + (batch.label if batch.label is not None else [])


class CollectorBase:
    def collect_gluon(self, name, _, arr):
        raise NotImplementedError()

    def collect_module(self, name, arr):
        name = mx.base.py_str(name)
        handle = ctypes.cast(arr, mx.base.NDArrayHandle)
        arr = mx.ndarray.NDArray(handle, writable=False)
        self.collect_gluon(name, '', arr)

    def pre_batch(self, m, b):
        pass

    def post_batch(self, m, b):
        pass


class CalibCollector(CollectorBase):
    """ Histogram and min/max collector """

    def __init__(self, include_tensors_kl, include_tensors_minmax, num_bins=8001):
        self.min_max_dict = {}
        self.hist_dict = {}
        self.num_bins = num_bins
        self.include_tensors_minmax = include_tensors_minmax
        self.include_tensors_kl = include_tensors_kl

    def collect_gluon(self, name, _, arr):
        if name in self.include_tensors_kl:
            alg = 'kl'
        elif name in self.include_tensors_minmax:
            alg = 'minmax'
        else:
            return

        min_range = mx.ndarray.min(arr).asscalar()  # pylint: disable=no-member
        max_range = mx.ndarray.max(arr).asscalar()  # pylint: disable=no-member
        th = max(abs(min_range), abs(max_range))
        # minmax (always)
        if name in self.min_max_dict:
            cur_min_max = self.min_max_dict[name]
            self.min_max_dict[name] = (min(cur_min_max[0], min_range),
                                       max(cur_min_max[1], max_range))
        else:
            self.min_max_dict[name] = (min_range, max_range)

        if alg == 'kl':  # histogram only when kl is specified
            arr = arr.asnumpy()
            if name in self.hist_dict:
                self.hist_dict[name] = _combine_histogram(self.hist_dict[name], arr,
                                                          min_range, max_range, th)
            else:
                hist, hist_edges = np.histogram(arr, bins=self.num_bins, range=(-th, th))
                self.hist_dict[name] = (hist, hist_edges, min_range, max_range, th)

    def calc_kl_th_dict(self, quantized_dtype, logger):
        if len(self.hist_dict) > 0:
            return _get_kl_thresholds(self.hist_dict, quantized_dtype, logger)
        return {}

    # for mxnet version >= 2.0.0
    def post_collect(self):
        return self.min_max_dict


class TensorCollector(CollectorBase):
    """ Tensors collector. Builds up qtensor_to_tensor mapping """

    def __init__(self, include_nodes, qtensor_to_tensor, tensor_to_node):
        self.tensors_dicts = []
        self.include_nodes = dict(include_nodes)
        self.qtensor_to_tensor = qtensor_to_tensor
        self.tensor_to_node = tensor_to_node

        rest = set(self.include_nodes.keys()) - set(self.tensor_to_node.values())
        assert len(rest) == 0, 'Unexpected tensors set to be collected: {}'.format(rest)

    def collect_gluon(self, name, _, arr):
        if name not in self.tensor_to_node:
            if name in self.qtensor_to_tensor:
                name = self.qtensor_to_tensor[name]
            else:
                qname, name = name, _qtensor_to_tensor(name, self.tensor_to_node)
                self.qtensor_to_tensor[qname] = name
            if name == '':
                return

        node = self.tensor_to_node[name]
        if node in self.include_nodes:
            op = self.include_nodes[node]
            key = (node, op)
            # allow rewriting only for quantize nodes
            assert (key not in self.tensors_dicts[-1] or name not in self.tensors_dicts[-1][key]
                    or op == QUANTIZE_OP_NAME)
            self.tensors_dicts[-1].setdefault(key, {})[name] = arr.copy()

    def pre_batch(self, m, b):
        self.tensors_dicts.append({})


class NameCollector(CollectorBase):
    def __init__(self):
        self.names = []

    def collect_gluon(self, name, _, arr):
        self.names.append(name)

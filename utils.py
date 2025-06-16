import os
from model.SGNN import *
import numpy as np
import random
import torch
from sklearn.cluster import KMeans, SpectralClustering
from metric import cal_clustering_metric
import scipy.sparse as sp
from sklearn.metrics import f1_score
import json
import argparse
import logging
from itertools import product
import re
import copy

def generate_overlooked_adjacency(adjacency, rate=0.0):
    """
    Generate the overlooked matrix. 
    The ignored entries are marked as 1. The others are 0. 

    Require:
        adjacency: a scipy sparse matrix. 
        rate: rate of overlooked entries, a float from 0 to 1. 
    Return: 
        overlooked_adjacency: a sparse n * n integer matrix whose entries are 0/1 .
    """
    # build sparse overlook matrix except for A_ii

    # --- old version: overlook unseen entries as well ---
    # overlook_matrix = sp.rand(size, size, density=rate, format='coo')
    # sparse_size = overlook_matrix.data.shape[0]  # num of ignored entries
    # sparse_data = np.ones(sparse_size)
    # overlook_matrix.data = sparse_data

    # --- only overlook edges --- 
    rate = min(max(rate, 0), 1)
    adj = adjacency.tocoo()
    size = adj.shape[0]
    sparse_size = adj.data.shape[0]
    mask_size = int(rate * sparse_size)
    idx = np.random.permutation(list(range(sparse_size)))
    idx = idx[:mask_size]
    row = adj.row[idx]
    col = adj.col[idx]
    data = np.ones(mask_size)
    overlook_matrix = sp.coo_matrix((data, (row, col)), shape=(size, size))
    overlook_matrix = overlook_matrix.maximum(overlook_matrix.transpose())

    # build self-loop to overlook reconstructions of A_ii
    idx = list(range(size))
    self_loop = sp.coo_matrix((np.ones(size), (idx, idx)), shape=(size, size))
    overlook_matrix = overlook_matrix.maximum(self_loop)
    return overlook_matrix


def csr_to_sparse_Tensor(csr_mat, device):
    coo_mat = csr_mat.tocoo()
    return coo_to_sparse_Tensor(coo_mat, device)


def coo_to_sparse_Tensor(coo_mat, device):
    idx = torch.LongTensor(np.vstack((coo_mat.row, coo_mat.col)))
    tensor = torch.sparse.IntTensor(idx, torch.FloatTensor(coo_mat.data), torch.Size(coo_mat.shape))
    return tensor.to(device)


def get_weight_initial(param_shape):
    bound = np.sqrt(6.0 / (param_shape[0] + param_shape[1]))
    ini = torch.rand(param_shape) * 2 * bound - bound
    return torch.nn.Parameter(ini, requires_grad=True)


def get_Laplacian_from_adjacency(adjacency):
    adj = adjacency + torch.eye(adjacency.shape)
    degree = torch.sum(adj, dim=1).pow(-0.5)
    return (adj * degree).t() * degree


def process_data_with_adjacency(adjacency, X, device):
    return process_data_with_adjacency_high_order(adjacency, X, device, order=1)


def process_data_with_adjacency_high_order(adjacency, X, device, order=1):
    size = X.shape[0]
    idx = list(range(size))
    idx = torch.LongTensor(np.vstack((idx, idx)))
    self_loop = torch.sparse.FloatTensor(idx, torch.ones(size), torch.Size((size, size))).to(device)
    adj = adjacency + self_loop
    # idx.minimum(1)

    # Compute sqrt(D)
    degree = torch.sparse.sum(adj, dim=1).to_dense().sqrt()
    # Compute D^(-1/2)
    degree = 1 / degree
    # X or H depending on current layer
    processed_X = X
    for i in range(order):
        # Computes D^(-1/2) * H
        processed_X = (processed_X.t() * degree).t()
        # Computes A * H
        processed_X = adj.mm(processed_X)
        # Computes D^(-1/2) * H
        processed_X = (processed_X.t() * degree).t()
    return processed_X


def k_means(embedding, n_clusters, labels, replicates=1):
    acc, nmi = (0, 0)
    for i in range(replicates):
        km = KMeans(n_clusters=n_clusters).fit(embedding)
        prediction = km.predict(embedding)
        a, n = cal_clustering_metric(labels, prediction)
        acc += a
        nmi += n
    return acc / replicates, nmi / replicates


def spectral_clustering(affinity, n_clusters, labels):
    spectralClustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    prediction = spectralClustering.fit_predict(affinity)
    acc, nmi = cal_clustering_metric(labels, prediction)
    return acc, nmi


def relaxed_k_means(X, n_clusters, labels):
    U, _, __ = torch.svd(X)
    indicator = U[:, :n_clusters]  # c-top
    indicator = indicator.detach()
    epsilon = torch.tensor(10 ** -7).to(X.device)
    indicator = indicator / indicator.norm(dim=1).reshape((indicator.shape[0], -1)).max(epsilon)
    indicator = indicator.detach().cpu().numpy()
    km = KMeans(n_clusters=n_clusters).fit(indicator)
    prediction = km.predict(indicator)
    acc, nmi = cal_clustering_metric(labels, prediction)
    return acc, nmi


def print_SGNN_info(stackedGNN, logger=None):
    def count_layers_and_log(layers):
        total_count = 0
        for i, layer in enumerate(layers):
            if isinstance(layer, list):
                sublayer_count = len(layer)
                logger.info(f"{i + 1}-th layer: {sublayer_count} sublayers of type [{layer[0]}]")
                total_count += sublayer_count
            else:
                logger.info(f"{i + 1}-th layer: {layer}")
                total_count += 1
        return total_count

    logger.info('\n============ Settings ============')
    total_layers = count_layers_and_log(stackedGNN.layers)
    logger.info('Totally {} layers:'.format(total_layers))
    logger.info('overlook_rates={}'.format(stackedGNN.overlooked_rates))
    logger.info('BP_count={}, eta={}\n'.format(stackedGNN.BP_count, stackedGNN.eta))



def clustering(X, labels):
    n_clusters = np.unique(labels).shape[0]
    acc, nmi = k_means(X, n_clusters, labels, replicates=5)
    print('k-means results: ACC: %5.4f, NMI: %5.4f' % (acc, nmi))


def clustering_tensor(X, labels, relaxed_kmeans=False):
    clustering(X.cpu().detach().numpy(), labels)
    n_clusters = np.unique(labels).shape[0]
    if not relaxed_kmeans:
        return
    rkm_acc, rkm_nmi = relaxed_k_means(X, n_clusters, labels)
    print('Relaxed K-Means results: ACC: %5.4f, NMI: %5.4f' % (rkm_acc, rkm_nmi))
    return rkm_acc, rkm_nmi
    # K = embedding.matmul(embedding.t()).abs()
    # K = (K + K.t()) / 2
    # affinity = K.cpu().detach().numpy()
    # sc_acc, sc_nmi = spectral_clustering(affinity, n_clusters, labels)
    # print('SC results: ACC: %5.4f, NMI: %5.4f' % (sc_acc, sc_nmi))


def classification(prediction, labels, mask=None, logger=None, debug=False):
    # num = labels.shape[0]
    # acc = (prediction == labels).sum() / num
    gnd = labels if mask is None else labels[mask]
    pred = prediction if mask is None else prediction[mask]
    acc = f1_score(gnd, pred, average='micro')
    f1 = f1_score(gnd, pred, average='macro')
    if debug:
        logger.debug('ACC: %5.4f, F1-Score: %5.4f' % (acc, f1))
    else:
        logger.info('ACC: %5.4f, F1-Score: %5.4f' % (acc, f1))
    return acc


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_hyperparam_combinations(base_config, params, tuning_params=None):
    """
    Generate all possible hyperparameter combinations and merge them with the base configuration.
    """
    param_grid = {}

    # Build the parameter grid for Cartesian product
    if tuning_params:
        for param in tuning_params:
            if param in params:
                param_grid[param] = params[param]
            elif param == "layers":
                layer_combinations = []
                for layer in params["layer"]:
                    layer_combinations.extend(product(*[layer[key] for key in layer]))
                param_grid["layers"] = layer_combinations
    else:
        param_grid = params

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    # Merge each combination with the base configuration
    merged_configs = []
    for combination in combinations:
        config_copy = base_config.copy()
        for key, value in combination.items():
            if key == "layers":
                # Replace layers in the base configuration
                config_copy["layers"] = []
                for layer_values in value:
                    layer_config = {k: v for k, v in zip(params["layer"][0].keys(), layer_values)}
                    config_copy["layers"].append(layer_config)
            else:
                config_copy[key] = value
        merged_configs.append(config_copy)

    return merged_configs


def set_arg_parser():
    ALLOWED_DATASETS = [
        "Cora",
        "Citeseer",
        "PubMed",
        "Flickr",
        "FacebookPagePage",
        "Actor",
        "LastFMAsia",
        "DeezerEurope",
        "Amazon Computers",
        "Amazon Photo",
        "Reddit",
        "Arxiv",
        "Products",
        "Mag",
        "Yelp"
    ]
    ALLOWED_MODELS = [
        "SGNN",
        "GCN",
        "SGC"
    ]
    ALLOWED_TASK_TYPES = [
        "Classification",
        "Clustering"
    ]

    parser = argparse.ArgumentParser(description="SGNN script")
    parser.add_argument("--cuda_num", type=str, required=True, help="GPU to use")
    parser.add_argument(
        "--model",
        type=str,
        choices=ALLOWED_MODELS,
        required=True,
        help=f"Model name (choices: {', '.join(ALLOWED_MODELS)})"
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=ALLOWED_DATASETS,
        required=True,
        help=f"Dataset name (choices: {', '.join(ALLOWED_DATASETS)})"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=ALLOWED_TASK_TYPES,
        required=True,
        help=f"Experiment type (choices: {', '.join(ALLOWED_TASK_TYPES)})"
    )
    parser.add_argument("--exp", type=int, required=True, help="How many times do you want to run the exercise")
    parser.add_argument("--log_path", type=str, help="Where you want to store the logs")
    parser.add_argument("--tuning", type=int, help="How many times you want to tune the hyperparameters")
    parser.add_argument("--ddp", action="store_true", default=False, help="Use Distributed Data Parallelism")
    parser.add_argument("--mm_op", default=None, type=str, help="Concat or Add")
    parser.add_argument("--mm_structure", default=None, type=str, help="1-2-1, 2-2-1, etc")
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["info", "debug"],
        default="info",
        help="Set the logging level (default: info)"
    )
    parser.add_argument(
        "--exps_file",
        type=str,
        required=False,
        default=None,
        help="Path to JSON file containing multiple experiment configurations to run sequentially."
    )
    args = parser.parse_args()

    return (args.cuda_num, args.data, args.model, args.task, args.exp, args.log_path,
            args.tuning, args.ddp, args.mm_op, args.mm_structure, args.log_level, args.exps_file)



class CustomFormatter(logging.Formatter):
    """Custom formatter to include the current GPU in log messages with colors."""

    # ANSI color codes
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Log format with GPU info
    format = "%(asctime)s - %(gpu_info)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    # Different colors for different log levels
    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        # Get current GPU info
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_id)
            record.gpu_info = f"GPU: {gpu_id} ({gpu_name})"
        else:
            record.gpu_info = "GPU: CPU"

        # Select the appropriate format based on log level
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_logger():
    """Sets up the logger with GPU info and color-coded formatting."""
    with open("global_settings.json", "r") as file:
        loaded_data = json.load(file)

    logger_settings = loaded_data["logger"]
    model = logger_settings["model"]
    log_path = logger_settings["log_path"]
    dataset_name = logger_settings["dataset"]
    log_level = logger_settings["log_level"]

    if log_path == "local":
        logs_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        log_path = f"{logs_dir}//{model}Concat_{dataset_name}.log"

    logger = logging.getLogger(model)

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(CustomFormatter())

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatter())

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Set the logger level dynamically
        logger.setLevel(logging.DEBUG if log_level == "DEBUG" else logging.INFO)

    return logger

def get_ddp_setting():
    with open("global_settings.json", "r") as file:
        return json.load(file)["ddp"]

def get_layer_type(layer_name: str):

    if layer_name == "GAE":
        return LayerParam.GAE
    elif layer_name == "GCN":
        return LayerParam.GCN
    elif layer_name == "EGCN":
        return LayerParam.EGCN
    elif layer_name == "SGC":
        return LayerParam.SGC
    else:
        raise ValueError(f"Unknown layer type: {layer_name}. Supported types are: GAE, GCN, EGCN, SGC.")

def construct_sgnn_layers(layer_config, is_large, lam):

    layers = []

    for layer in layer_config:

        if isinstance(layer, list):
            sub_layers = construct_sgnn_layers(layer, is_large, lam)
            layers.append(sub_layers)

        else:
            current_layer_activation = layer["activation"]
            current_layer_inner_act = layer["inner_act"]
            try:
                current_layer_type = layer["layer_type"]
            except KeyError:
                current_layer_type = "EGCN"

            chosen_act = get_activation(current_layer_activation)
            chosen_inner_act = get_activation(current_layer_inner_act)
            chosen_layer_type = get_layer_type(current_layer_type)

            if is_large:
                layer_to_add = LayerParam(layer["neurons"], inner_act=chosen_inner_act, act=chosen_act,
                                          gnn_type=chosen_layer_type,
                                          learning_rate=layer["learning_rate"],
                                          max_iter=layer["max_iter"], lam=lam, batch_size=layer["batch_size"])
            else:
                layer_to_add = LayerParam(layer["neurons"], inner_act=chosen_inner_act, act=chosen_act,
                                          gnn_type=chosen_layer_type,
                                          learning_rate=layer["learning_rate"],
                                          order=layer["order"], max_iter=layer["max_iter"],
                                          lam=lam, batch_size=layer["batch_size"])

            layers.append(layer_to_add)

    return layers


def parse_structure(structure):
    """
    Parse a GNN structure string like "SGC-(SGC-SGC)-GCN" into a nested list representation.
    Handles shorthand like "2SGC" or "(8EGCN)" and expands them appropriately.

    :param structure: Structure string.
    :return: Parsed structure as a nested list.
    """
    import re

    def expand_shorthand(layer_str):
        """
        Expand shorthand like '2SGC' or '8EGCN' to their full form.
        """
        match = re.match(r'(\d+)([A-Z]+)', layer_str)
        if match:
            count, layer_type = match.groups()
            return '-'.join([layer_type] * int(count))
        return layer_str

    # Expand shorthands like "2SGC" or "(8EGCN)"
    structure = re.sub(r'(\d+)([A-Z]+)', lambda m: expand_shorthand(m.group(0)), structure)

    # Match patterns in the structure
    groups = re.findall(r'\(([^()]+)\)|(\w+)', structure)

    parsed = []
    for group in groups:
        if group[0]:  # If it's inside parentheses
            parsed.append(group[0].split('-'))
        elif group[1]:  # If it's a single layer
            parsed.append(group[1])
    return parsed

def generate_layers(parsed_struct, original_layers):
    """
    Generate new layers list based on parsed structure.

    Logic:
    - The first *single* layer uses original layer 0
    - Any nested list duplicates original layer 0 configs
    - The last *single* layer uses the last original layer config
    """
    new_layers = []
    n = len(original_layers)

    def recursive_build(struct, depth=0):
        result = []
        for i, elem in enumerate(struct):
            if isinstance(elem, list):
                # Nested list - duplicate original_layers[0]
                duplicated_layers = [
                    copy.deepcopy(original_layers[0]) for _ in range(len(elem))
                ]
                # Set each layer_type accordingly from the elem
                for j, lt in enumerate(elem):
                    duplicated_layers[j]['layer_type'] = lt
                result.append(duplicated_layers)
            else:
                # Single element
                # Determine if this is first or last single layer in top-level structure
                if depth == 0:
                    # At top level, check position for first or last
                    if i == 0:
                        base_layer = copy.deepcopy(original_layers[0])
                    elif i == len(struct) - 1:
                        base_layer = copy.deepcopy(original_layers[-1])
                    else:
                        # Middle single elements - fallback to first layer config for safety
                        base_layer = copy.deepcopy(original_layers[0])
                else:
                    # Nested single elements (if any) just copy original layer 0
                    base_layer = copy.deepcopy(original_layers[0])

                base_layer['layer_type'] = elem
                result.append(base_layer)
        return result

    new_layers = recursive_build(parsed_struct, depth=0)
    return new_layers


def modify_and_return_config(dataset_config, structure, operation_type):
    """
    Modify the dataset configuration to match the specified GNN structure and operation type.
    Handles nested structures and ensures correct base layer usage.

    - For 1-1-1 to 1-2-1: Duplicate the second layer into a list.
    - For 1-1 to 1-2-1: Duplicate the first layer into a list for the second layer.
    """
    parsed = parse_structure(structure)  # Parse the structure into a nested list
    original_layers = dataset_config['layers']

    new_layers = []

    for i, layer_type in enumerate(parsed):
        if isinstance(layer_type, list):  # Handle parenthesis groups
            # If intermediate group, duplicate either second or first layer depending on the original structure
            if len(original_layers) >= 3:
                base_layer = copy.deepcopy(original_layers[1])  # Use the second layer as base
            else:
                base_layer = copy.deepcopy(original_layers[0])  # Use the first layer as base

            list_of_layers = []
            for sub_layer_type in layer_type:
                layer = copy.deepcopy(base_layer)
                layer['layer_type'] = sub_layer_type
                list_of_layers.append(layer)
            new_layers.append(list_of_layers)
        else:  # Handle single layers
            if i == 0:  # First layer
                base_layer = copy.deepcopy(original_layers[0])
            elif i == len(parsed) - 1:  # Last layer
                base_layer = copy.deepcopy(original_layers[-1])
            else:  # Intermediate layers
                if len(original_layers) >= 3:
                    base_layer = copy.deepcopy(original_layers[1])  # Use the second layer as base
                else:
                    base_layer = copy.deepcopy(original_layers[0])  # Use the first layer as base
            base_layer['layer_type'] = layer_type
            new_layers.append(base_layer)

    # Update the dataset configuration
    dataset_config['layers'] = new_layers
    if operation_type:
        dataset_config['operation'] = operation_type

    return dataset_config


def get_activation(current_layer_activation):

    if "tanh" in current_layer_activation:
        chosen_activation = Func(torch.nn.functional.tanh)
    elif "sigmoid" in current_layer_activation:
        chosen_activation = Func(torch.nn.functional.sigmoid)
    elif "linear" in current_layer_activation:
        chosen_activation = Func(None)
    elif "leaky" in current_layer_activation:
        negative_slope = float(current_layer_activation.split("=")[1])
        chosen_activation = Func(torch.nn.functional.leaky_relu, negative_slope=negative_slope)
    elif current_layer_activation == "relu":
        chosen_activation = Func(torch.nn.functional.relu)
    else:
        print("Not activation type set")
        exit()

    return chosen_activation

class Func(torch.nn.Module):
    def __init__(self, func, **params):
        super(Func, self).__init__()
        self.func = func
        self.params = params

    def forward(self, X):
        return X if self.func is None else self.func(X, **self.params)

    def __repr__(self):
        s = '{}'.format('<linear' if self.func is None else self.func)
        s = Func.process_func_name(s)
        if self.params:
            s = s + ' with ' + str(self.params)
        return s

    @staticmethod
    def process_func_name(s):
        s = s[1:]
        s = s.split(' at')[0]
        return '<{}>'.format(s)

class LayerParam:
    GAE = 0
    GCN = 1
    EGCN = 2
    SGC = 3
    MASK_RATE = 'mask_rate'

    def __init__(self, neurons, inner_act, act, gnn_type, **kwargs):
        self.neurons = neurons
        self.inner_activation = inner_act
        self.activation = act
        self.gnn_type = gnn_type
        self.extra_params = kwargs

    def __repr__(self) -> str:
        s = 'Neurons: {}, inner_activation: {}, activations: {} '
        s = s.format(self.neurons, self.inner_activation, self.activation)
        if self.extra_params:
            st = ' with parameters: {} '.format(self.extra_params)
            s += st
        st = 'error!'
        if self.gnn_type == LayerParam.GAE:
            st = 'type: GAE'
        elif self.gnn_type == LayerParam.GCN:
            st = 'type: GCN'
        elif self.gnn_type == LayerParam.EGCN:
            st = 'type: EGCN'
        elif self.gnn_type == LayerParam.SGC:
            st = 'type: SGC'
        return s + st

    def get(self, key, default):
        return self.extra_params.get(key, default)
import os

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
    degree = torch.sparse.sum(adj, dim=1).to_dense().sqrt()
    degree = 1 / degree

    processed_X = X
    for i in range(order):
        processed_X = (processed_X.t() * degree).t()
        processed_X = adj.mm(processed_X)
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
    logger.info('\n============ Settings ============')
    logger.info('Totally {} layers:'.format(len(stackedGNN.layers)))
    for i, layer in enumerate(stackedGNN.layers):
        logger.info('{}-th layer: {}'.format(i + 1, layer))
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

def sample_hyperparams(filename, dataset_name):
    """Reads hyperparameter ranges from a JSON file and randomly selects a configuration."""
    random.seed()

    with open(filename, "r") as f:
        data = json.load(f)

    params = data["Test"]

    # Randomly sample values for global hyperparameters
    sampled_params = {
        "eta": random.choice(params["eta"]),
        "BP_count": random.choice(params["BP_count"]),
        "lam": random.choice(params["lam"]),
        "layers": []
    }

    # Determine random number of layers (2 or 3)
    num_layers = random.choice([2, 3])

    # Sample values for each layer dynamically
    for _ in range(num_layers):
        sampled_layer = {
            "neurons": random.choice(params["layer"][0]["neurons"]),
            "inner_act": random.choice(params["layer"][0]["inner_act"]),
            "activation": random.choice(params["layer"][0]["activation"]),
            "learning_rate": random.choice(params["layer"][0]["learning_rate"]),
            "order": random.choice(params["layer"][0]["order"]),
            "max_iter": random.choice(params["layer"][0]["max_iter"]),
            "batch_size": random.choice(params["layer"][0]["batch_size"])
        }
        sampled_params["layers"].append(sampled_layer)

    return sampled_params

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

    parser = argparse.ArgumentParser(description="SGNN script")
    parser.add_argument("--cuda_num", type=str, required=True, help="GPU to use")
    parser.add_argument(
        "--model",
        type=str,
        choices=ALLOWED_MODELS,  # Restricts choices
        required=True,
        help=f"Model name (choices: {', '.join(ALLOWED_MODELS)})"
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=ALLOWED_DATASETS,  # Restricts choices
        required=True,
        help=f"Dataset name (choices: {', '.join(ALLOWED_DATASETS)})"
    )
    parser.add_argument("--task", type=str, required=True, help="Classification or Clustering")
    parser.add_argument("--exp", type=int, required=True, help="How many times do you want to run the exercise")
    parser.add_argument("--log_path", type=str, help="Where you want to store the logs")
    parser.add_argument("--tuning", type=int, help="How many times you want to tune the hyperparameters")
    parser.add_argument("--ddp", action="store_true", default=False, help="Use Distributed Data Parallelism")
    args = parser.parse_args()

    cuda_num = args.cuda_num
    dataset_decision = args.data
    model_decision = args.model
    task_type = args.task
    exp_times = args.exp
    log_path = args.log_path
    is_tuning = args.tuning
    ddp = args.ddp

    return cuda_num, dataset_decision, model_decision, task_type, exp_times, log_path, is_tuning, ddp


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

    if log_path == "local":
        logs_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

    log_path = f"{logs_dir}//{model}_{dataset_name}.log"

    logger = logging.getLogger(model)

    # Check if handlers already exist (to prevent duplication)
    if not logger.handlers:
        # File handler (logs to file)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(CustomFormatter())

        # Console handler (prints to console)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatter())

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.setLevel(logging.DEBUG)

    return logger

def get_ddp_setting():
    with open("global_settings.json", "r") as file:
        return json.load(file)["ddp"]
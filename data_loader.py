import numpy as np
import scipy.sparse as sp
import scipy.io as scio
import pickle as pkl
import networkx as nx
import sys
from ogb.nodeproppred import PygNodePropPredDataset
import json
import torch
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.datasets import (Planetoid, Reddit, Flickr, FacebookPagePage, Actor, LastFMAsia, DeezerEurope,
                                      Amazon, Yelp)
from torch_geometric.utils import add_self_loops

YALE = 'Yale'
UMIST = 'UMIST'
THREE_RINGS = 'three_rings'


def get_training_data(dataset_choice):

    if dataset_choice == "Cora" or dataset_choice == "Citeseer" or dataset_choice == "PubMed":
        data = load_planetoid_dataset(dataset_choice)
    elif dataset_choice == "Flickr":
        data = load_flickr_data()
    elif dataset_choice == "FacebookPagePage":
        data = load_facebook_pagepage_dataset()
    elif dataset_choice == "Actor":
        data = load_actor_dataset()
    elif dataset_choice == "LastFMAsia":
        data = load_lastfmasia_dataset()
    elif dataset_choice == "DeezerEurope":
        data = load_deezereurope_dataset()
    elif dataset_choice == "Amazon Computers":
        data = load_amazon_dataset(dataset_choice.split(" ")[1])
    elif dataset_choice == "Amazon Photo":
        data = load_amazon_dataset(dataset_choice.split(" ")[1])
    elif dataset_choice == "Reddit":
        data = load_reddit_data()
    elif dataset_choice == "Yelp":
        data = load_yelp_data()
    elif dataset_choice == "Arxiv":
        data = load_ogbn_dataset(dataset_choice.lower())
    elif dataset_choice == "Products":
        data = load_ogbn_dataset(dataset_choice.lower())
    elif dataset_choice == "Mag":
        data = load_ogbn_dataset(dataset_choice.lower())
    else:
        print("Invalid dataset")
        exit()

    return data


def load_reddit_from_npz(dataset_dir: str):
    # Load NPZ files for NPZ-based Reddit data
    adj = sp.load_npz(dataset_dir + "reddit_adj.npz")
    data = np.load(dataset_dir + "reddit.npz")
    return (adj, data['feats'], data['y_train'], data['y_val'],
            data['y_test'], data['train_index'], data['val_index'], data['test_index'])


def load_reddit_data(dataset_dir: str = "data/") -> Data:

    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = load_reddit_from_npz(dataset_dir)

    num_nodes = adj.shape[0]
    labels = np.zeros(num_nodes)
    labels[train_index] = y_train
    labels[val_index] = y_val
    labels[test_index] = y_test

    # Convert adjacency matrix to edge_index format
    adj = adj + adj.T  # ensure symmetry
    edge_index, _ = from_scipy_sparse_matrix(adj)

    # Normalize features
    features = torch.FloatTensor(features)
    features = (features - features.mean(dim=0)) / features.std(dim=0)

    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(np.isin(np.arange(num_nodes), train_index))
    val_mask = torch.BoolTensor(np.isin(np.arange(num_nodes), val_index))
    test_mask = torch.BoolTensor(np.isin(np.arange(num_nodes), test_index))

    # Load built-in Reddit dataset for reference (do not use for training here)
    pyg_data = Reddit(root='./data/Reddit')
    num_features = pyg_data.num_features
    num_classes = pyg_data.num_classes

    data = Data(x=features, y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, adjacency=adj,
                num_features=num_features, num_classes=num_classes, edge_index=edge_index)
    return data


def load_ogbn_dataset(dataset_n):

    dataset_name = f'ogbn-{dataset_n}'
    dataset = PygNodePropPredDataset(name=dataset_name, root='data/')
    data = dataset[0]  # For OGBN-MAG, this is a HeteroData object
    split_idx = dataset.get_idx_split()

    if dataset_n.lower() == 'mag':
        # 1) Features & labels
        features = data.x_dict['paper']  # shape [num_papers, num_features]
        labels = data.y_dict['paper'].view(-1)  # shape [num_papers]

        # 2) Number of paper nodes
        num_nodes = features.size(0)  # or features.shape[0]

        # 3) Citation edges among papers
        edge_index = data.edge_index_dict[('paper', 'cites', 'paper')]

        # 4) Train/val/test splits for "paper" nodes
        train_index = split_idx['train']['paper']
        val_index = split_idx['valid']['paper']
        test_index = split_idx['test']['paper']
    else:
        # --- Homogeneous access for Arxiv/Products ---
        features = data.x.numpy()
        labels = data.y.squeeze().numpy()
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        train_index = split_idx['train']
        val_index = split_idx['valid']
        test_index = split_idx['test']

    # Convert edge_index to a SciPy sparse adjacency matrix:
    adjacency = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    data = Data(x=features, y=labels, train_mask=train_index, val_mask=val_index, test_mask=test_index,
                adjacency=adjacency, num_features=dataset.num_features, num_classes=dataset.num_classes,
                edge_index=edge_index)

    return data


def load_flickr_data():

    pyg_data = Flickr(root=f"data/Flickr")

    # Paths to the raw Flickr data
    raw_dir = f"./data/Flickr/raw/"

    # Load data
    adj_full = sp.load_npz(raw_dir + "adj_full.npz")
    features = np.load(raw_dir + "feats.npy")
    with open(raw_dir + "class_map.json") as f:
        class_map = json.load(f)
    with open(raw_dir + "role.json") as f:
        roles = json.load(f)

    # Convert class_map to labels
    labels = np.array([class_map[str(i)] for i in range(len(class_map))])

    # Create train, val, and test masks
    num_nodes = len(labels)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    train_mask[roles["tr"]] = True
    val_mask[roles["va"]] = True
    test_mask[roles["te"]] = True

    # Create adjacency matrix
    adj = nx.adjacency_matrix(nx.from_scipy_sparse_array(adj_full))

    data = Data(x=features, y=labels, train_mask=train_mask, val_mask=val_mask,
                test_mask=test_mask, adjacency=adj, pyg_data=pyg_data)

    return data


def load_yelp_data():

    # Load Yelp dataset
    dataset = Yelp(root='data/Yelp')

    # Select the first (and only) graph in the dataset
    data = dataset[0]

    # Convert multi-label to single-label by taking the dominant label
    labels = data.y.argmax(dim=1)
    unique_classes = torch.unique(labels)
    mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_classes)}
    labels = torch.tensor([mapping[label.item()] for label in labels], dtype=torch.long)

    # Add self-loops (SGC requires this)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)

    edge_index = data.edge_index

    # Extract masks
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    # Convert adjacency to sparse matrix
    adjacency = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

    features = data.x

    num_features = dataset.num_node_features
    num_classes = dataset.num_classes

    data = Data(x=features, y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, adjacency=adjacency
                , num_features=num_features, num_classes=num_classes, edge_index=edge_index)

    return data


def load_facebook_pagepage_dataset():

    pyg_data = FacebookPagePage(root=f"data/FacebookPagePage")

    # Load the raw data
    data = np.load(f"data/FacebookPagePage/raw/facebook.npz", allow_pickle=True)

    # Extract edges, features, and target (labels)
    edges = data["edges"]  # Edge list
    features = data["features"]  # Node features
    labels = data["target"]  # Node labels

    # Create adjacency matrix from edge list
    num_nodes = features.shape[0]
    adjacency = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    ).tocsc()

    # Assuming no train/val/test masks in this dataset, split manually (if needed)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    # Example: First 70% for training, next 15% for validation, last 15% for testing
    train_mask[: int(0.7 * num_nodes)] = True
    val_mask[int(0.7 * num_nodes): int(0.85 * num_nodes)] = True
    test_mask[int(0.85 * num_nodes):] = True

    data = Data(x=features, y=labels, train_mask=train_mask, val_mask=val_mask,
                test_mask=test_mask, adjacency=adjacency, pyg_data=pyg_data)

    return data


def load_lastfmasia_dataset():

    pyg_data = LastFMAsia(root=f"data/LastFMAsia")

    # Load the raw data
    data = np.load(f"data/LastFMAsia/raw/lastfm_asia.npz", allow_pickle=True)

    # Extract edges, features, and target (labels)
    edges = data["edges"]  # Edge list
    features = data["features"]  # Node features
    labels = data["target"]  # Node labels

    # Create adjacency matrix from edge list
    num_nodes = features.shape[0]
    adjacency = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    ).tocsc()

    # Assuming no train/val/test masks in this dataset, split manually (if needed)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    # Example: First 70% for training, next 15% for validation, last 15% for testing
    train_mask[: int(0.7 * num_nodes)] = True
    val_mask[int(0.7 * num_nodes): int(0.85 * num_nodes)] = True
    test_mask[int(0.85 * num_nodes):] = True

    data = Data(x=features, y=labels, train_mask=train_mask, val_mask=val_mask,
                test_mask=test_mask, adjacency=adjacency, pyg_data=pyg_data)

    return data


def load_deezereurope_dataset():

    pyg_data = LastFMAsia(root=f"data/DeezerEurope")

    # Load the raw data
    data = np.load(f"data/DeezerEurope/raw/deezer_europe.npz", allow_pickle=True)

    # Extract edges, features, and target (labels)
    edges = data["edges"]  # Edge list
    features = data["features"]  # Node features
    labels = data["target"]  # Node labels

    # Create adjacency matrix from edge list
    num_nodes = features.shape[0]
    adjacency = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    ).tocsc()

    # Assuming no train/val/test masks in this dataset, split manually (if needed)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    # Example: First 70% for training, next 15% for validation, last 15% for testing
    train_mask[: int(0.7 * num_nodes)] = True
    val_mask[int(0.7 * num_nodes): int(0.85 * num_nodes)] = True
    test_mask[int(0.85 * num_nodes):] = True

    data = Data(x=features, y=labels, train_mask=train_mask, val_mask=val_mask,
                test_mask=test_mask, adjacency=adjacency, pyg_data=pyg_data)

    return data

def load_actor_dataset():

    data = Actor(root=f"data/Actor")

    # File paths
    edges_file = "data/Actor/raw/out1_graph_edges.txt"
    features_labels_file = "data/Actor/raw/out1_node_feature_label.txt"

    # Load edges (tab-separated values)
    edges = np.loadtxt(edges_file, dtype=int, delimiter="\t", skiprows=1)

    # Process features and labels manually due to mixed delimiter
    features = []
    labels = []
    max_feature_length = 0  # Track the maximum length of feature vectors

    with open(features_labels_file, "r") as f:
        lines = f.readlines()[1:]  # Skip the header line
        for line in lines:
            parts = line.strip().split("\t")  # Split by tab
            feature_values = list(map(float, parts[1].split(",")))  # Split features by comma
            label = int(parts[-1])  # Last column is the label

            features.append(feature_values)
            labels.append(label)

            # Update maximum feature length
            max_feature_length = max(max_feature_length, len(feature_values))

    # Pad features to the maximum feature length
    padded_features = np.zeros((len(features), max_feature_length), dtype=np.float32)
    for i, feature_row in enumerate(features):
        padded_features[i, :len(feature_row)] = feature_row

    # Convert to PyTorch tensors
    features = torch.tensor(padded_features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)  # Ensure long dtype for classification

    # Create adjacency matrix
    num_nodes = len(features)
    adjacency = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )

    # Convert adjacency matrix to PyG format
    edge_index, _ = from_scipy_sparse_matrix(adjacency)

    # Generate train/validation/test masks manually
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Example: First 70% for training, next 15% for validation, last 15% for testing
    train_mask[: int(0.7 * num_nodes)] = True
    val_mask[int(0.7 * num_nodes): int(0.85 * num_nodes)] = True
    test_mask[int(0.85 * num_nodes):] = True

    # Create PyG Data object
    data_obj = Data(
        x=features,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        adjacency=adjacency,
        num_features=data.num_features,
        num_classes=data.num_classes,  # Number of unique classes
        edge_index=edge_index
    )

    return data_obj



def load_amazon_dataset(dataset_type):

    pyg_data = Amazon(root=f"data/Amazon", name=dataset_type)

    # Load the raw data
    data = np.load(f"data/Amazon/{dataset_type}/raw/amazon_electronics_{dataset_type.lower()}.npz", allow_pickle=True)

    # Extract adjacency matrix components
    adj_data = data["adj_data"]
    adj_indices = data["adj_indices"]
    adj_indptr = data["adj_indptr"]
    adj_shape = tuple(data["adj_shape"])
    adjacency = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)

    # Extract features
    attr_data = data["attr_data"]
    attr_indices = data["attr_indices"]
    attr_indptr = data["attr_indptr"]
    attr_shape = tuple(data["attr_shape"])
    features = sp.csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape).todense()  # Ensure dense matrix

    # Extract labels
    labels = np.array(data["labels"], dtype=np.int64)  # Convert labels to int64 for compatibility

    # Create train, validation, and test masks
    num_nodes = features.shape[0]
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    # Example split: First 70% for training, next 15% for validation, last 15% for testing
    train_mask[: int(0.7 * num_nodes)] = True
    val_mask[int(0.7 * num_nodes): int(0.85 * num_nodes)] = True
    test_mask[int(0.85 * num_nodes):] = True

    features = np.array(features, dtype=np.float32)

    data = Data(x=features, y=labels, train_mask=train_mask, val_mask=val_mask,
                test_mask=test_mask, adjacency=adjacency, pyg_data=pyg_data)

    return data
def load_cora():
    path = 'data/cora/'
    data_name = 'cora'
    print('Loading from raw data file...')
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, data_name), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    _, _, labels = np.unique(idx_features_labels[:, -1], return_index=True, return_inverse=True)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, data_name), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj.T + adj
    adj = adj.minimum(1)
    return features, idx_map, adj, labels


def load_citeseer():
    path = 'data/citeseer/'
    data_name = 'citeseer'
    print('Loading from raw data file...')
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, data_name), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    _, _, labels = np.unique(idx_features_labels[:, -1], return_index=True, return_inverse=True)

    idx = np.array(idx_features_labels[:, 0], dtype=np.str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, data_name), dtype=np.str)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    rows_to_delete = []
    for i in range(edges_unordered.shape[0]):
        if edges[i, 0] is None or edges[i, 1] is None:
            rows_to_delete.append(i)
    edges = np.array(np.delete(edges, rows_to_delete, axis=0), dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj.T + adj
    adj = adj.minimum(1)
    return features, idx_map, adj, labels


def load_pubmed():
    print('Loading from raw data file...')
    data = scio.loadmat('data/pubmed.mat')
    adj = data['W']
    # adj = adj - adj.diagonal()
    features = data['fea']
    # adj = sp.coo_matrix(adj)
    labels = data['gnd']
    labels = np.reshape(labels, (labels.shape[0],))
    adj = adj.T + adj
    adj = adj.minimum(1)
    return features, adj.tocoo(), labels


def load_citeseer_from_mat():
    print('Loading from raw data file...')
    data = scio.loadmat('data/citeseer.mat')
    adj = data['W']
    features = data['fea']
    labels = data['gnd']
    labels = np.reshape(labels, (labels.shape[0],))

    # Convert adjacency to sparse matrix
    adj = sp.coo_matrix(adj)

    # Ensure symmetry and convert to binary (0 or 1)
    adj = adj + adj.T
    adj = adj.minimum(1)

    return features, adj.tocsr(), labels


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)

def load_planetoid_dataset(name):
    # Load dataset
    dataset = Planetoid(root=f"./data/{name}", name=name)
    data = dataset[0]  # Planetoid datasets contain only one graph



    # Extract features, labels, and masks
    features = data.x
    labels = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    edge_index = data.edge_index  # Ensure this is in the correct shape

    num_nodes = features.shape[0]

    edges = data.edge_index.numpy().T
    adjacency = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    ).tocsc()

    # Create the final Data object
    data_obj = Data(
        x=features,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        adjacency=adjacency,  # Extra: adjacency matrix
        num_features=features.shape[1],
        num_classes=labels.max().item() + 1,
        edge_index=edge_index
    )

    return data_obj

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/node/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/node/ind.{}.test.index".format(dataset_str.lower()))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'Citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    features = features.toarray()
    # label = (y_train + y_val + y_test).argmax(axis=1)
    label = labels.argmax(axis=1)
    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    num_classes = labels.shape[1]
    num_features = features.shape[1]
    row, col = adj.nonzero()  # gets the indices of non-zero entries
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)

    data = Data(x=features, y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, adjacency=adj
                , num_features=num_features, num_classes=num_classes, edge_index=edge_index)

    return data


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

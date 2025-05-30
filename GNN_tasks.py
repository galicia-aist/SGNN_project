from model.SGC import SGC, train, evaluate
from model.SGNN import *
from data_loader import get_training_data
import warnings
from datetime import datetime
from torch_geometric.loader import NeighborLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_logger, get_ddp_setting

warnings.filterwarnings('ignore')
utils.set_seed(0)


def run_classificaton_with_SGNN(rank, world_size, dataset_choice, config, return_queue):

    accuracy = None

    ddp = get_ddp_setting()

    if ddp:
        ddp_setup(rank, world_size)

    logger = get_logger()

    start_time = datetime.now()

    data = get_training_data(dataset_choice)

    labels = data.y
    features = data.x
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    adjacency = data.adjacency

    n_class = np.unique(labels).shape[0]
    if (type(features) is not np.ndarray and dataset_choice != "Reddit" and dataset_choice != "Mag"
            and dataset_choice != "Yelp"):
        features = features.to_dense()
    features = torch.Tensor(features)

    if torch.cuda.is_available():
        if ddp:
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device(f"cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device(f"cpu")

    # ========== training setting ==========
    eta = config["eta"]
    BP_count = config["BP_count"]
    is_large = config["isLarge"]

    features = features.to(device)
    layer_config = config["layers"]
    lam = config["lam"]

    try:
        operation_type = config["operation"]
    except (KeyError, TypeError):
        operation_type = None


    # ========== layers setting ==========

    layers = utils.construct_sgnn_layers(layer_config, is_large, lam)

    # ========== overlook setting ==========
    overlook_rates = None

    sgnn = SupervisedStackedGNN(features, adjacency, layers,
                                training_mask=train_mask, val_mask=test_mask,
                                overlooked_rates=overlook_rates,
                                BP_count=BP_count, eta=eta, device=device,
                                labels=labels, metric_func=utils.classification, logger=logger, rank=rank,
                                mmop=operation_type)

    if ddp and rank == 0 or not ddp:
        utils.print_SGNN_info(sgnn, logger=logger)

    logger.debug('============ Start Training ============')
    prediction = sgnn.run()
    logger.debug('============ End Training ============')

    if ddp:
        dist.barrier()

    if ddp and rank == 0 or not ddp:

        # ========== Testing ==========
        logger.info('============ Start testing ============')
        logger.info("Training accuracy")
        utils.classification(prediction, labels, train_mask, logger=logger)
        logger.info("Validation accuracy")
        utils.classification(prediction, labels, val_mask, logger=logger)
        logger.info("Test accuracy")
        accuracy = utils.classification(prediction, labels, test_mask, logger=logger)

    finish_time = datetime.now()

    time_difference = finish_time - start_time

    # Extract hours, minutes, and seconds
    total_seconds = int(time_difference.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(start_time.strftime("Process started at: " + "%Y-%m-%d %H:%M:%S"))
    logger.info(finish_time.strftime("Process started at: " + "%Y-%m-%d %H:%M:%S"))
    logger.info(f"Training lasted {hours} hours, {minutes} minutes, {seconds} seconds")
    total_max_iter = 0
    total_max_sub_iter = 0
    for layer in layer_config:
        if isinstance(layer, list):
            for sub_layer in layer:
                total_max_sub_iter = total_max_sub_iter + sub_layer["max_iter"]
            total_max_iter = total_max_iter + total_max_sub_iter
        else:
            total_max_iter = total_max_iter + layer["max_iter"]

    total_iterations = total_max_iter*((BP_count*2)+1)
    logger.info(f"Total iterations: {total_iterations}")
    efficiency = total_seconds / total_iterations
    logger.info(f"Official efficiency: {efficiency}")

    if ddp:
        if rank == 0:
            return_queue.put((accuracy, efficiency, total_seconds))

        dist.destroy_process_group()
    else:
        return accuracy, efficiency, total_seconds, total_iterations


def run_classification_with_SGC(rank, world_size, dataset_choice, config, return_queue):

    accuracy = None
    ddp = get_ddp_setting()

    if ddp:
        ddp_setup(rank, world_size)

    start_time = datetime.now()
    is_large = config["isLarge"]
    epochs = config["epochs"]
    batch_size_train = None
    batch_size_test = None
    if is_large:
        batch_size_train = config["batch_size_train"]
        batch_size_test = config["batch_size_test"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]

    if torch.cuda.is_available():
        if ddp:
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device(f"cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device(f"cpu")

    logger = get_logger()

    data = get_training_data(dataset_choice)

    if is_large:
        if ddp:
            train_sampler = DistributedSampler(
                data.train_mask.nonzero().squeeze(),  # Get only training indices
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )

            test_sampler = DistributedSampler(
                data.test_mask.nonzero().squeeze(),  # Get only test indices
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )

            train_loader = NeighborLoader(
                data,
                num_neighbors=[10, 10],  # Sample 10 neighbors per layer
                batch_size=batch_size_train,
                input_nodes=data.train_mask,
                sampler=train_sampler  # Ensure each GPU only gets a part of the dataset
            )

            test_loader = NeighborLoader(
                data,
                num_neighbors=[10, 10],
                batch_size=batch_size_test,
                input_nodes=data.test_mask,
                sampler=test_sampler  # Ensure correct test distribution
            )
        else:
            train_loader = NeighborLoader(
                data,
                num_neighbors=[10, 10],  # Sample 10 neighbors per layer
                batch_size=batch_size_train,
                input_nodes=data.train_mask
            )

            test_loader = NeighborLoader(
                data,
                num_neighbors=[10, 10],
                batch_size=batch_size_test,
                input_nodes=data.test_mask
            )
    else:
        train_loader = None
        test_loader = None

    if is_large:
        pass
    else:
        data = data.to(device)

    model = SGC(data).to(device)
    if ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(1, epochs):
        loss = train(model, optimizer, device, data, train_loader=train_loader, dataset_name=dataset_choice)
        logger.debug(f'Epoch {epoch}: Loss: {loss:.4f}')

    # Synchronize before evaluation
    if ddp:
        dist.barrier()
        if rank == 0:
            accuracy = evaluate(model, device, data, test_loader=test_loader, dataset_name=dataset_choice)
            logger.info(f"Test Accuracy: {accuracy:.4f}")
    else:
        accuracy = evaluate(model, device, data, test_loader=test_loader, dataset_name=dataset_choice)
        logger.info(f"Test Accuracy: {accuracy:.4f}")

    finish_time = datetime.now()
    time_difference = finish_time - start_time

    logger.info(start_time.strftime("Process started at: %Y-%m-%d %H:%M:%S"))
    logger.info(finish_time.strftime("Process finished at: %Y-%m-%d %H:%M:%S"))

    total_seconds = int(time_difference.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(f"Training lasted {hours} hours, {minutes} minutes, {seconds} seconds")

    total_iterations = epochs
    efficiency = total_seconds / total_iterations
    logger.info(f"Official efficiency: {efficiency}")

    if ddp:
        if rank == 0:
            return_queue.put((accuracy, efficiency, total_seconds))

        dist.destroy_process_group()
    else:
        return accuracy, efficiency, total_seconds, total_iterations


def run_clustering_with_SGNN(dataset_choice, config):
    dataset_name = None
    start_time = datetime.now()

    adjacency, features, labels, train_mask, val_mask, test_mask  = get_training_data(dataset_choice)

    mask_rate = config["mask_rate"]
    overlook_rates = config["overlook_rates"]
    layers = config["layers"]
    max_iter = config["max_iter"]
    batch_size = config["batch"]
    BP_count = config["BP_count"]
    learning_rate = ["learning_rate"]
    lam = config["lam"]
    eta = config["eta"]
    loss = config["loss"]
    negative_slope = config["negative_slope"]

    # ========== load data ==========
    # features, _, adjacency, labels = load_cora()
    # features, _, adjacency, labels = load_citeseer()
    # features, adjacency, labels = load_citeseer_from_mat()
    # features, adjacency, labels = load_pubmed()
    n_clusters = np.unique(labels).shape[0]
    if type(features) is not np.ndarray:
        features = features.todense()
    features = torch.Tensor(features)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ========== training setting ==========
    features = features.to(device)

    # ========== layers setting ==========
    # layers = [32, 16]
    # layers = [128, 64, 32]  # Cora
    # layers = [256, 128]

    relu_func = Func(torch.nn.functional.relu)
    linear_func = Func(None)
    leaky_relu_func = Func(torch.nn.functional.leaky_relu, negative_slope=negative_slope)

    if (dataset_name == "PubMed" or dataset_name == "Citeseer" or dataset_name == "Flickr"
            or dataset_name == "FacebookPagePage" or dataset_name == "Actor" or dataset_name == "LastFMAsia"
            or dataset_name == "DeezerEurope" or dataset_name == "Amazon"):
        layers = [
            LayerParam(layers[0], inner_act=linear_func, act=leaky_relu_func, gnn_type=LayerParam.GAE,
                       mask_rate=mask_rate, lam=lam, max_iter=max_iter, learning_rate=learning_rate,
                       batch_size=batch_size),
            LayerParam(layers[1], inner_act=linear_func, act=linear_func, gnn_type=LayerParam.GAE,
                       mask_rate=mask_rate, lam=lam, max_iter=max_iter, learning_rate=learning_rate,
                       batch_size=batch_size),
        ]
    else:
        layers = [
            LayerParam(layers[0], inner_act=linear_func, act=leaky_relu_func, gnn_type=LayerParam.GAE,
                       mask_rate=mask_rate, lam=lam, max_iter=max_iter, learning_rate=learning_rate,
                       batch_size=batch_size),
            LayerParam(layers[1], inner_act=linear_func, act=leaky_relu_func, gnn_type=LayerParam.GAE,
                       mask_rate=mask_rate, lam=lam, max_iter=max_iter, learning_rate=learning_rate,
                       batch_size=batch_size),
            LayerParam(layers[2], inner_act=linear_func, act=linear_func, gnn_type=LayerParam.GAE,
                       mask_rate=mask_rate, lam=lam, max_iter=max_iter, learning_rate=learning_rate,
                       batch_size=batch_size),
        ]

    # ========== overlook setting ==========
    overlook_rates = None

    sgae = StackedGNN(features, adjacency, layers,
                      overlooked_rates=overlook_rates, BP_count=BP_count,
                      eta=eta, device=device,
                      labels=labels, metric_func=utils.clustering, logger=logger)

    utils.print_SGNN_info(sgae)

    print('============ Start Training ============')
    embedding = sgae.run()
    print('============ End Training ============')

    utils.print_SGNN_info(sgae)

    # ========== Clustering ==========
    print('============ Start Clustering ============')
    accuracy, nmi = utils.clustering_tensor(embedding.detach(), labels, relaxed_kmeans=True)
    finish_time = datetime.now()
    print(start_time.strftime("Process started at: " + "%Y-%m-%d %H:%M:%S"))
    print(finish_time.strftime("Process finished at: " + "%Y-%m-%d %H:%M:%S"))
    time_difference = finish_time - start_time

    # Extract hours, minutes, and seconds
    total_seconds = int(time_difference.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Training lasted {hours} hours, {minutes} minutes, {seconds} seconds")

    total_iterations = max_iter * len(layers) * ((BP_count * 2) + 1)
    print(f"Total iterations: {total_iterations}")
    efficiency = total_seconds / total_iterations
    print(f"Official efficiency: {efficiency}")

    return accuracy, efficiency, nmi, dataset_name




def ddp_setup(rank, world_size):
    """Train function for distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
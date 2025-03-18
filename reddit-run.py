import torch
from model import *
from data_loader import *
import warnings
from input_data import load_data
from reddit_utils import load_graphsage_data
from datetime import datetime

warnings.filterwarnings('ignore')

decision = input("Choose which dataset to use\n1. Reddit\n2. Arxiv\n3. Products\nYour input: ")
dataset_name = None

if decision == "1":
    dataset_name = "Reddit"
    num_data, _, full_adj, feats, _, _, labels, _, _, _ = load_graphsage_data('reddit')
elif decision == "2":
    dataset_name = "arxiv"
    full_adj, num_data, feats, labels, _, _, _ = load_ogbn_dataset(dataset_name)
elif decision == "3":
    dataset_name = "products"
    full_adj, num_data, feats, labels, _, _, _ = load_ogbn_dataset(dataset_name)
else:
    print("Invalid")
    exit()

# ========== load data ==========
# num_data, _, full_adj, feats, _, _, labels, _, _, _ = load_graphsage_data('reddit')
# full_adj, num_data, feats, labels, _, _, _ = load_ogbn_arxiv()

_ = None
adjacency = full_adj
n_clusters = np.unique(labels).shape[0]
features = torch.Tensor(feats)
feats = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
features = features.to(device)

# ========== training setting ==========

learning_rate = 10**-4
# max_iter = 250000
max_iter = 10000

batch_size = 512


# ========== layers setting ==========
# layers = [128, 64]
relu_func = Func(torch.nn.functional.relu)
linear_func = Func(None)
leaky_relu_func = Func(torch.nn.functional.leaky_relu, negative_slope=0.2)


lam = 10**-6

layers = [
    LayerParam(128, inner_act=leaky_relu_func, act=linear_func, gnn_type=LayerParam.GAE,
               mask_rate=0.2, learning_rate=learning_rate, lam=lam, max_iter=max_iter, batch_size=batch_size),
    LayerParam(64, inner_act=leaky_relu_func, act=linear_func, gnn_type=LayerParam.GAE,
               mask_rate=0.2, learning_rate=learning_rate, lam=lam, max_iter=max_iter, batch_size=batch_size),
]


# ========== overlook setting ==========
overlook_rates = None

sgae = StackedGNN(features, adjacency, layers,
                  overlooked_rates=overlook_rates, BP_count=5,
                  eta=10, device=device, labels=labels, metric_func=utils.clustering)

utils.print_SGNN_info(sgae)
start_time = datetime.now()
print('============ Start Training ============')
embedding = sgae.run()
print('============ End Training ============')

utils.print_SGNN_info(sgae)

# ========== Clustering ==========
print('============ Start Clustering ============')
utils.clustering(embedding.cpu().detach().numpy(), labels)

finish_time = datetime.now()
print(start_time.strftime("Process started at: " + "%Y-%m-%d %H:%M:%S"))
print(finish_time.strftime("Process started at: " + "%Y-%m-%d %H:%M:%S"))
time_difference = finish_time - start_time

# Extract hours, minutes, and seconds
total_seconds = int(time_difference.total_seconds())
hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Training lasted {hours} hours, {minutes} minutes, {seconds} seconds")

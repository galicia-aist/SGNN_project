import torch
from torch_geometric.nn import SGConv
import torch.nn.functional as F
from torch.nn import Linear


# Define the SGC model

class SGC(torch.nn.Module):
    def __init__(self, data):
        super(SGC, self).__init__()
        hidden_dim = 64  # Specify the hidden dimension for the intermediate layer
        self.conv1 = SGConv(data.num_features, hidden_dim, K=2)  # First layer
        self.conv2 = SGConv(hidden_dim, data.num_classes, K=2)  # Second layer

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)  # Add non-linearity after the first layer
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


# Training function
def train(model, optimizer, device, data, train_loader=None, dataset_name=None):
    model.train()
    total_loss = 0
    if train_loader is None:
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss
    else:
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            out = model(batch)
            if dataset_name == 'Arxiv' or dataset_name == 'Mag' or dataset_name == 'Products':
                loss = F.nll_loss(out, batch.y)
            else:
                loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])  # Now `batch.y` is 1D
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)


# Evaluation function
def evaluate(model, device, data, test_loader=None, dataset_name=''):
    model.eval()
    correct = 0
    total = 0
    if test_loader is None:
        out = model(data)
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        acc = correct / data.test_mask.sum().item()
        return acc
    else:
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                pred = out.argmax(dim=1)
                if dataset_name == 'Arxiv' or dataset_name == 'Mag' or dataset_name == 'Products':
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
                else:
                    correct += (pred[batch.test_mask] == batch.y[batch.test_mask]).sum().item()
                    total += batch.test_mask.sum().item()
        return correct / total

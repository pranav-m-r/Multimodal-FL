# task.py
"""multimodal: A Flower / PyTorch app."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
import pandas as pd

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset


class SplitAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(6, 3, 5),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class CanonicallyCorrelatedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_A = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.encoder_B = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.decoder_A = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.decoder_B = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
        )

    def forward(self, input_A, input_B):
        h_A = self.encoder_A(input_A)
        h_B = self.encoder_B(input_B)
        recon_A = self.decoder_A(h_A)
        recon_B = self.decoder_B(h_B)
        return h_A, h_B, recon_A, recon_B

    def loss(self, input_A, input_B, recon_A, recon_B, h_A, h_B):
        recon_loss = F.mse_loss(recon_A, input_A) + F.mse_loss(recon_B, input_B)
        correlation_loss = -torch.trace(torch.matmul(h_A.T, h_B))
        return recon_loss + correlation_loss

def train_multimodal(model, trainloader, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        for batch in trainloader:
            input_A, input_B = batch["input_A"].to(device), batch["input_B"].to(device)
            h_A, h_B, recon_A, recon_B = model(input_A, input_B)
            loss = model.loss(input_A, input_B, recon_A, recon_B, h_A, h_B)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss.item()

def test_multimodal(model, testloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in testloader:
            input_A, input_B = batch["input_A"].to(device), batch["input_B"].to(device)
            h_A, h_B, recon_A, recon_B = model(input_A, input_B)
            loss = model.loss(input_A, input_B, recon_A, recon_B, h_A, h_B)
            total_loss += loss.item()
    return total_loss / len(testloader), None

def multimodal_fedavg(results, multimodal_weight):
    num_clients = len(results)
    agg_parameters = None

    for _, (parameters, num_samples, metadata) in enumerate(results):
        weight = num_samples
        if metadata.get("multimodal", False):
            weight *= multimodal_weight

        scaled_params = [param * weight for param in parameters]

        if agg_parameters is None:
            agg_parameters = scaled_params
        else:
            agg_parameters = [agg_param + scaled_param for agg_param, scaled_param in zip(agg_parameters, scaled_params)]

    total_weight = sum(
        num_samples * (multimodal_weight if metadata.get("multimodal", False) else 1)
        for _, num_samples, metadata in results
    )

    return [param / total_weight for param in agg_parameters]

def load_opp_data(partition_id, num_partitions):
    """Load Opportunity dataset."""
    data_path = "path_to_opportunity_dataset"

    # Load and preprocess dataset (modify path as necessary)
    raw_data = pd.read_csv(f"{data_path}/partition_{partition_id}.csv")
    features = raw_data.iloc[:, :-1].values.astype(np.float32)  # Features
    labels = raw_data.iloc[:, -1].values.astype(np.int64)  # Labels

    # Split features and labels into train/test
    train_size = int(0.8 * len(features))
    train_features, test_features = features[:train_size], features[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]

    train_data = [{"input_A": f, "input_B": f, "label": l} for f, l in zip(train_features, train_labels)]
    test_data = [{"input_A": f, "input_B": f, "label": l} for f, l in zip(test_features, test_labels)]

    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = DataLoader(test_data, batch_size=32)
    return trainloader, testloader

def load_data(partition_id, num_partitions, multimodal=False):
    if multimodal:
        return load_opp_data(partition_id, num_partitions)
    else:
        global fds
        if fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10", partitioners={"train": partitioner}
            )

        partition = fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        def apply_transforms(batch):
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch

        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
        testloader = DataLoader(partition_train_test["test"], batch_size=32)
        return trainloader, testloader

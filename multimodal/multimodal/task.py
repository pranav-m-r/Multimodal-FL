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


def train(model, trainloader, epochs, device):
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


def test(model, testloader, device):
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

        # Check if the result contains multimodal data (using metadata)
        if metadata.get("multimodal", False):
            weight *= multimodal_weight

        # Assuming the model has 'input_A' and 'input_B', you might need to scale them differently
        # based on the modality (e.g., A, B, or AB)

        # Aggregate modality-specific parameters
        scaled_params = [param * weight for param in parameters]

        if agg_parameters is None:
            agg_parameters = scaled_params
        else:
            agg_parameters = [
                agg_param + scaled_param
                for agg_param, scaled_param in zip(agg_parameters, scaled_params)
            ]

    # Calculate the total weight based on the number of samples and the multimodal weight
    total_weight = sum(
        num_samples * (multimodal_weight if metadata.get("multimodal", False) else 1)
        for _, num_samples, metadata in results
    )

    # Normalize aggregated parameters by total weight
    return [param / total_weight for param in agg_parameters]


def load_data(partition_id, num_partitions):
    """Load Opportunity dataset from .pt files."""
    data_path = "data"

    # Load the tensors from the .pt files
    train_data = torch.load(f"{data_path}/train{partition_id}.pt")
    test_data = torch.load(f"{data_path}/test{partition_id}.pt")

    # Assuming the data is in the form (features, labels)
    train_features = train_data[:, :-1].float()  # All but the last column as features
    train_labels = train_data[:, -1].long()  # Last column as labels
    test_features = test_data[:, :-1].float()
    test_labels = test_data[:, -1].long()

    # Create the data in the required format
    train_data = [
        {"input_A": f, "input_B": f, "label": l}
        for f, l in zip(train_features, train_labels)
    ]
    test_data = [
        {"input_A": f, "input_B": f, "label": l}
        for f, l in zip(test_features, test_labels)
    ]

    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = DataLoader(test_data, batch_size=32)

    return trainloader, testloader


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

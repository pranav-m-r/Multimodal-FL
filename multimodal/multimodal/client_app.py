"""multimodal: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from multimodal.task import (
    Net,
    get_weights,
    load_data,
    set_weights,
    test,
    train,
    SplitAutoencoder,
    CanonicallyCorrelatedAutoencoder,
    train_multimodal,
    test_multimodal,
)


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, valloader, local_epochs, multimodal=False):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.multimodal = multimodal
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        if self.multimodal:
            train_loss = train_multimodal(
                self.model, self.trainloader, self.local_epochs, self.device
            )
        else:
            train_loss = train(
                self.model, self.trainloader, self.local_epochs, self.device
            )
        return (
            get_weights(self.model),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        if self.multimodal:
            loss, accuracy = test_multimodal(self.model, self.valloader, self.device)
        else:
            loss, accuracy = test(self.model, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    multimodal = context.node_config.get("multimodal", False)
    if multimodal:
        model = (
            SplitAutoencoder()
        )  # Use correlated autoencoder for multimodal
    else:
        model = Net()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions, multimodal)
    local_epochs = context.run_config["local-epochs"]

    return FlowerClient(
        model, trainloader, valloader, local_epochs, multimodal
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)

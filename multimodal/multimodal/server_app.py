"""multimodal: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy
from multimodal.task import Net, get_weights, multimodal_fedavg


# Define Multimodal FedAvg Strategy
class MultimodalFedAvg(Strategy):
    def __init__(self, multimodal_weight=100):
        super().__init__()
        self.multimodal_weight = multimodal_weight

    def aggregate_fit(self, rnd, results, failures):
        return multimodal_fedavg(results, self.multimodal_weight)


def server_fn(context: Context):
    # Load model and initialize parameters
    num_rounds = context.run_config["num-server-rounds"]
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Configure strategy and server settings
    strategy = MultimodalFedAvg()
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)

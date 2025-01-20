"""multimodal: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy
from multimodal.task import Net, get_weights, multimodal_fedavg, set_weights, test


# Define Multimodal FedAvg Strategy
class MultimodalFedAvg(Strategy):
    def __init__(self, multimodal_weight=100):
        super().__init__()
        self.multimodal_weight = multimodal_weight

    def aggregate_fit(self, rnd, results, failures):
        """Aggregates the results from the clients."""
        
        # Call the multimodal_fedavg function, which handles the multimodal aggregation
        aggregated_parameters = multimodal_fedavg(results, self.multimodal_weight)
        return aggregated_parameters

    def aggregate_evaluate(self, rnd, results, failures):
        """Aggregates evaluation results from the clients."""
        # You can add any aggregation logic here for evaluation metrics, e.g., accuracy
        return {}

    def configure_fit(self, rnd, parameters, client_manager):
        """Configures the fit process, specifying the number of epochs or other settings."""
        config = {"local-epochs": 5}  # Customize this as needed
        return config

    def configure_evaluate(self, rnd, parameters, client_manager):
        """Configures the evaluation process."""
        config = {}  # Customize if you have specific configurations for evaluation
        return config

    def initialize_parameters(self, client_manager):
        """Initialize parameters for all clients."""
        # You can initialize parameters here if needed (e.g., random initialization)
        return []  # This can be a list of initial parameters if you have one

    def evaluate(self, parameters, config, client_manager):
        """Evaluates the global model after aggregation on test data."""
        # Assuming you have access to some global test dataset or clients for evaluation
        model = Net()  # Or load the actual global model

        # Set the parameters for the global model
        set_weights(model, parameters)

        # For simplicity, assuming all clients can evaluate the model
        total_loss = 0
        total_accuracy = 0
        num_samples = 0
        
        # Get the parameters as numpy arrays, this should match Flower's internal format
        parameters = parameters_to_ndarrays(parameters)
        
        # For now, let's simulate the evaluation with dummy data
        for client in client_manager.all():
            # Here we need to ensure we're using the correct test data
            loss, accuracy = test(model, client.get_test_data(), "cpu")  # Adjust to your test method
            total_loss += loss * accuracy
            total_accuracy += accuracy * accuracy
            num_samples += accuracy

        # You can return the average loss/accuracy as global evaluation
        return total_loss / num_samples, {"accuracy": total_accuracy / num_samples}


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

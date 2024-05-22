import flwr as fl
from flwr.server import start_server
from omegaconf import DictConfig
from centralized import test
from collections import OrderedDict
import torch
from hydra.utils import instantiate

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):

        #if server_round >50:
        #   lr =config.lr / 10
        return {"lr": config.lr,"momentum":config.momentum,
                "local_epochs":config.local_epochs}
    
    return fit_config_fn

def get_evalulate_fn(model_cfg: int, testloader):
    def evaluate_fn(server_round: int, parameters, config):

        model = instantiate(model_cfg)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader)

        return loss, {"accuracy":accuracy}

# Legacy mode
if __name__ == "__main__":
    start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=fl.server.strategy.FedAvg(),
    )

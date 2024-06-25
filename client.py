import pickle
from pathlib import Path

import flwr as fl
import os
from collections import OrderedDict
from centralized import load_model, train, test, Net
from typing import Callable, Dict, List, Tuple
import torch
import pandas as pd
import data_loading
from flwr.common import NDArrays, Scalar
from omegaconf import DictConfig
from hydra.utils import instantiate

from torch.utils.data import DataLoader

class FlowerClient(fl.client.NumPyClient):
  def __init__(self,
               trainloader,
               valloader,
               model_cfg) -> None:
     super().__init__()

     self.trainloader = trainloader
     self.valloader = valloader
     self.model = instantiate(model_cfg)

     self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  #サーバーから受け取ったパラメータでローカルモデルの重みを更新します
  def set_parameters(self, parameters: NDArrays):
   params_dict = zip(self.model.state_dict().keys(), parameters)
   state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
   self.model.load_state_dict(state_dict, strict=True)
  
  #モデルの重みを NumPy ndarray のリストとして返します
  def get_parameters(self, config: Dict[str, Scalar]):
    return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

  #ローカルモデルの重みを設定する ローカルモデルをトレーニングする 更新されたローカル モデルの重みを受け取る
  def fit(self, parameters, config):
    self.set_parameters(parameters)

    lr = config["lr"]
    momentum = config["momentum"]
    epochs = config["local_epochs"]

    #SGD
    #optim = torch.optim.SGD(self.model.parameters(),lr=lr, momentum=momentum)
    #adam
    optim = torch.optim.Adam(self.model.parameters(),lr=lr)

    train(self.model, self.trainloader,optim,epochs)
    return self.get_parameters({}), len(self.trainloader), {}
  
  #ローカルモデルをテストする
  def evaluate(self, parameters:NDArrays, config: Dict[str, Scalar]):
      
      self.set_parameters(parameters)
      loss, mae_loss, mse_loss, rmse_loss = test(self.model, self.valloader)
      #print("vall_mse_loss:", mse_loss)
      return float(loss), len(self.valloader), {"mae_loss": float(mae_loss)}

class FedBNFlowerClient(FlowerClient):
    """Similar to FlowerClient but this is used by FedBN clients."""

    def __init__(self, save_path: Path, client_id: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # For FedBN clients we need to persist the state of the BN
        # layers across rounds. In Simulation clients are statess
        # so everything not communicated to the server (as it is the
        # case as with params in BN layers of FedBN clients) is lost
        # once a client completes its training. An upcoming version of
        # Flower suports stateful clients
        bn_state_dir = save_path / "bn_states"
        bn_state_dir.mkdir(exist_ok=True)
        self.bn_state_pkl = bn_state_dir / f"client_{client_id}.pkl"

    def _save_bn_statedict(self) -> None:
        """Save contents of state_dict related to BN layers."""
        bn_state = {
            name: val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" in name
        }

        with open(self.bn_state_pkl, "wb") as handle:
            pickle.dump(bn_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_bn_statedict(self) -> Dict[str, torch.tensor]:
        """Load pickle with BN state_dict and return as dict."""
        with open(self.bn_state_pkl, "rb") as handle:
            data = pickle.load(handle)
        bn_stae_dict = {k: torch.tensor(v) for k, v in data.items()}
        return bn_stae_dict

    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays w or w/o using BN.

        layers.
        """
        # First update bn_state_dir
        self._save_bn_statedict()
        # Excluding parameters of BN layers when using FedBN
        return [
            val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" not in name
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays Exclude the bn layer if.

        available.
        """
        keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)

        # Now also load from bn_state_dir
        if self.bn_state_pkl.exists():  # It won't exist in the first round
            bn_state_dict = self._load_bn_statedict()
            self.model.load_state_dict(bn_state_dict, strict=False)


def generate_client_fn(trainloaders, valloaders, model_cfg): 
    def client_fn(cid: str): 
        client = FlowerClient(trainloader=trainloaders[int(cid)], 
                            valloader=valloaders[int(cid)], 
                            model_cfg=model_cfg,
                            )
        return client
        #return client.to_client() 

    return client_fn
      
# Start Flower client
#fl.client.start_numpy_client(
#  server_address="127.0.0.1:8080",
#  client=FlowerClient()
#  )





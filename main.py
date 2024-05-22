
import pickle
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl

from data_set import prepare_dataset_alldata, prepare_dataset_beta
from client import generate_client_fn
from server import get_evalulate_fn
from utils import quick_plot

@hydra.main(config_path = "conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    # 1. parse config  & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    # 2. prepare your dataset
    trainloaders, validationloaders, testloader = prepare_dataset_beta(cfg.num_clients,
                                                                   cfg.batch_size)
    #print(len(trainloaders),len(trainloaders[0].dataset))
    
    # 3. Define your clients
    client_fn =  generate_client_fn(trainloaders, validationloaders, cfg.model)


    # 4. Define your strategy
    strategy = instantiate(cfg.strategy, evaluate_fn=get_evalulate_fn(cfg.model, testloader))

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn = client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources ={"num_cpus":2, "num_gpus":0.0},
    )
    # 6. Save your results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results.pkl"

    results = {"history": history, "anythingelse":"here"}

    with open(str(results_path), "wb") as h:
        pickle.dump(results,h,protocol=pickle.HIGHEST_PROTOCOL)
    # Simple plot
    quick_plot(results_path)

if __name__ =="__main__":

    main()
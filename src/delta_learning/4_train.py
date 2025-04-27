import os
from os.path import join
import gc
import shutil
import socket
import logging
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def main():
    log.info("Running on host: " + str(socket.gethostname()))

    cfg = OmegaConf.load("params.yaml")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)

    inputs_train = torch.load(
        join('results', 'data', 'prepared_data', 'representations.pth'), 
        weights_only=True
    )['train']
    energies_train = torch.load(
        join('results', 'data', 'prepared_data', 'energies_train.pth'),
        weights_only=True
    )

    os.mkdir(join('results', 'model'))

    model = instantiate(cfg.model.init_function)

    if cfg.model.use_forces:
        forces_train = torch.load(
            join('results', 'data', 'prepared_data', 'forces_train.pth'), 
            weights_only=True
        )
        model.fit(inputs_train, energies_train, forces=forces_train)
    else:
        model.fit(inputs_train, energies_train)

if __name__ == '__main__':
    main()
 
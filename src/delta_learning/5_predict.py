import os
from os.path import  join
import logging
import socket
import inspect
import torch
import gpytorch
from hydra.utils import instantiate
from omegaconf import OmegaConf


log = logging.getLogger(__name__)

def main():
    log.info("Running on host: " + str(socket.gethostname()))

    cfg = OmegaConf.load("params.yaml")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)

    model = instantiate(cfg.model.init_function)

    os.mkdir(join('results', 'prediction'))
    with gpytorch.settings.max_cholesky_size(1000000), torch.no_grad(), gpytorch.settings.debug(False):
        if cfg.model.use_forces:
            make_prediction(model, 'al', cfg)
            make_prediction(model, 'test', cfg)
        else:
            make_prediction_without_forces(model, 'al', cfg)
            make_prediction_without_forces(model, 'test', cfg)


def make_prediction(model, part, cfg):

    inputs = torch.load(
        join('results', 'data', 'prepared_data', 'representations.pth'), 
        weights_only=True
    )[part]
    refs_e = torch.load(
        join('results', 'data', 'prepared_data', f'energies_{part}.pth'), 
        weights_only=True
    )[cfg.dataset.high_fidelity]
    refs_f = torch.load(
        join('results', 'data', 'prepared_data', f'forces_{part}.pth'),
        weights_only=True
    )[cfg.dataset.high_fidelity]

    refs_e_low = torch.load(
        join('results', 'data', 'prepared_data', f'energies_{part}.pth'), 
        weights_only=True
    )[cfg.dataset.low_fidelity]
    refs_f_low = torch.load(
        join('results', 'data', 'prepared_data', f'forces_{part}.pth'), 
        weights_only=True
    )[cfg.dataset.low_fidelity]

    if 'low_fidelity_energies' in inspect.signature(model.predict).parameters:
        preds_e, preds_f = model.predict(
            inputs, 
            low_fidelity_energies=refs_e_low, 
            low_fidelity_forces=refs_f_low
        )
    else:
        preds_e, preds_f = model.predict(inputs)

    if 'low_fidelity_energies' in inspect.signature(model.uncertainty).parameters:
        uncertainties_e, uncertainties_f = model.uncertainty(
            inputs,
            low_fidelity_energies=refs_e_low, 
            low_fidelity_forces=refs_f_low
        )
    else:
        uncertainties_e, uncertainties_f = model.uncertainty(inputs)

    torch.save(refs_e, join('results', 'prediction', f'refs_e_{part}'))
    torch.save(refs_f, join('results', 'prediction', f'refs_f_{part}'))
    torch.save(preds_e, join('results', 'prediction', f'preds_e_{part}'))
    torch.save(preds_f, join('results', 'prediction', f'preds_f_{part}'))
    torch.save(uncertainties_e, join('results', 'prediction', f'uncertainties_e_{part}'))
    torch.save(uncertainties_f, join('results', 'prediction', f'uncertainties_f_{part}'))


def make_prediction_without_forces(model, part, cfg):

    inputs = torch.load(
        join('results', 'data', 'prepared_data', 'representations.pth'), 
        weights_only=True
        )[part]
    refs_e = torch.load(
        join('results', 'data', 'prepared_data', f'energies_{part}.pth'), 
        weights_only=True
    )[cfg.dataset.high_fidelity]
    refs_low = torch.load(
        join('results', 'data', 'prepared_data', f'energies_{part}.pth'), 
        weights_only=True
    )[cfg.dataset.low_fidelity]

    if 'low_fidelity_targets' in inspect.signature(model.predict).parameters:
        preds_e = model.predict(inputs, low_fidelity_targets=refs_low)
    else:
        preds_e = model.predict(inputs)

    if 'low_fidelity_targets' in inspect.signature(model.uncertainty).parameters:
        uncertainties_e = model.uncertainty(inputs, low_fidelity_targets=refs_low)
    else:
        uncertainties_e = model.uncertainty(inputs)

    torch.save(refs_e, join('results', 'prediction', f'refs_e_{part}'))
    torch.save(preds_e, join('results', 'prediction', f'preds_e_{part}'))
    torch.save(uncertainties_e, join('results', 'prediction', f'uncertainties_e_{part}'))


if __name__ == '__main__':
    main()

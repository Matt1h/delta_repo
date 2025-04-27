import os
from os.path import  join
import socket
import logging
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np

import torch

log = logging.getLogger(__name__)


PART_NAMES = ['train', 'al', 'test']


def main():
    log.info("Running on host: " + str(socket.gethostname()))

    cfg = OmegaConf.load("params.yaml")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)

    os.mkdir(join('results', 'data', 'prepared_data'))

    # load subset
    subset_inds = instantiate(cfg.dataset.subset.subset_function)
    subset_inds = subset_inds if subset_inds is not None else slice(None)

    # calc representations
    path = f'../datasets/{cfg.dataset.path}'
    coordinates = instantiate(cfg.dataset.load_coordinates_function)(path)[subset_inds]
    n_configs = len(coordinates)
    representation_func = instantiate(cfg.representation.function)
    if representation_func is None:
        representations = torch.tensor(np.reshape(coordinates, (n_configs, -1)))
    else:
        charges = torch.tensor(instantiate(cfg.dataset.load_charges_function)(path))
        representations = representation_func(list(torch.tensor(coordinates).reshape(n_configs, -1)), charges=charges)

    # load properties
    energies = np.load(join('results', 'data', 'new_data', 'energies.npz'))
    forces = np.load(join('results', 'data', 'new_data', 'forces.npz'))

    # shuffle data
    rng = np.random.default_rng(cfg['s3_prepare_data']['seed'])
    shuffle_inds = rng.permutation(n_configs)

    # separate data
    inds = separate_(shuffle_inds, n_configs, cfg['s3_prepare_data']['n_train'], cfg['s3_prepare_data']['n_test'])
    torch.save(inds, join('results', 'data', 'prepared_data', 'inds.pth'))

    # store separated representations
    store_separated_representations(representations, inds)

    # remove offset
    offsets = None
    if cfg['s3_prepare_data']['remove_offset']:
        energies_train = extract_energy_part(energies, inds, 'train')
        offsets = {fidelity_name: values.mean() for fidelity_name, values in energies_train.items()}
        torch.save(offsets, join('results', 'data', 'prepared_data', 'offsets.pth'))

    # store separated properties
    store_separated_properties(energies, inds, offsets, 'energies')
    store_separated_properties(forces, inds, offsets, 'forces')



def store_separated_representations(representations, inds):
    representations_separated = {part_name: representations[inds[part_name]] for part_name in PART_NAMES}
    torch.save(representations_separated, join('results', 'data', 'prepared_data', 'representations.pth'))


def store_separated_properties(properties, inds, offsets, property_name):
    for part_name in PART_NAMES:
        if property_name == 'energies':
            energies_part = extract_energy_part(properties, inds, part_name, offsets=offsets)
            torch.save(energies_part, join('results', 'data', 'prepared_data', f'energies_{part_name}.pth'))
        if property_name == 'forces':
            forces_part = extract_forces_part(properties, inds, part_name)
            torch.save(forces_part, join('results', 'data', 'prepared_data', f'forces_{part_name}.pth'))


def extract_energy_part(energies, inds, part_name, offsets=None):
    energies_part = {}
    for fidelity_name, sf_energies in energies.items():
        sf_energies = torch.tensor(sf_energies[1][inds[part_name]])
        if offsets is not None:
            sf_energies = sf_energies - offsets[fidelity_name]
        energies_part[fidelity_name] = sf_energies

    return energies_part


def extract_forces_part(forces, inds, part_name):
    forces_part = {}
    for fidelity_name, sf_forces in forces.items():
        sf_forces = torch.tensor(sf_forces[inds[part_name]])
        forces_part[fidelity_name] = sf_forces
    return forces_part


def separate_(inputs, n, n_train, n_test):
    if n_test > 0:
        res = {
            'train': inputs[:n_train],
            'al': inputs[n_train:n - n_test],
            'test': inputs[-n_test:]
        }
        return res
    elif n_test == 0:
        res = {
            'train': inputs[:n_train],
            'al': inputs[n_train:n - n_test],
            'test': inputs[:0]
        }
        return res


if __name__ == '__main__':
    main()

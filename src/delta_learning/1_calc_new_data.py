import os
from os.path import  join
import socket
import logging
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np
from ase import Atoms
from xtb.ase.calculator import XTB

log = logging.getLogger(__name__)


def main():
    log.info("Running on host: " + str(socket.gethostname()))

    cfg = OmegaConf.load("params.yaml")

    subset_inds = instantiate(cfg.dataset.subset.subset_function)
    subset_inds = subset_inds if subset_inds is not None else slice(None)

    path = f'../datasets/{cfg.dataset.path}'
    coordinates = instantiate(cfg.dataset.load_coordinates_function)(path)[subset_inds]
    charges = instantiate(cfg.dataset.load_charges_function)(path)
    energies_full = instantiate(cfg.dataset.load_energies_function)(path)
    energies = {k: (np.arange(len(coordinates)), energies_full[k][1][subset_inds]) for k in energies_full.keys()}
    forces_full = instantiate(cfg.dataset.load_forces_function)(path)
    forces = {k: forces_full[k][1][subset_inds] for k in forces_full.keys()}

    # calc cheap labels
    energies_new_fidelity = []
    forces_new_fidelity = []
    for (i, single_frame_coordinates) in enumerate(coordinates):
        atoms = Atoms(positions=single_frame_coordinates, numbers=charges)

        atoms.calc = XTB()
        energies_new_fidelity.append(atoms.get_potential_energy())
        forces_new_fidelity.append(atoms.get_forces())

    energies['XTB'] = (np.arange(len(energies_new_fidelity)), energies_new_fidelity)
    forces['XTB'] = np.array(forces_new_fidelity)

    os.mkdir(join('results', 'data', 'new_data')) #
    np.savez(join('results', 'data', 'new_data', 'energies.npz'), **energies)
    np.savez(join('results', 'data', 'new_data', 'forces.npz'), **forces)

if __name__ == '__main__':
    main()

import os
from os.path import join
from typing import List, Dict
import shutil
import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from sgdml.train import GDMLTrain
from sgdml.predict import GDMLPredict
import subprocess


class SingleFidelitySGDML:
    def __init__(
            self, 
            fidelity,
            data_conversion_script_path=None,
            train_kwargs=None,
            task_kwargs=None,
            charges_function=None, 
            dataset_charges_path=None,
        ):
        self.fidelity = fidelity
        repo_dir = join(os.path.expanduser("~"), 'dev', 'delta_repo')  # TODO: other solution
        self.data_conversion_script_path = join(repo_dir, '..', data_conversion_script_path)
        self.train_kwargs = train_kwargs
        self.task_kwargs = task_kwargs
        dataset_charges_path = f'{repo_dir}/../datasets/{dataset_charges_path}'
        self.charges = charges_function(dataset_charges_path)

        self.model_dir = join('results', 'model')

    def fit(self, inputs, energies, forces=None):
        self.fit_single_fidelity_(inputs, energies, forces=forces)

    def predict(self, inputs):
        return self.predict_single_fidelity_(inputs)

    def uncertainty(self, inputs):
        return torch.rand(len(inputs)), torch.rand(inputs.shape[0], int(inputs.shape[1]/3), 3)

    def fit_single_fidelity_(self, inputs, energies, forces=None, fidelity=None):

        fidelity = fidelity or self.fidelity
        
        # convert data to extended xyz
        xyz_path = join(self.model_dir, f'train_dataset_{fidelity}.xyz')
        if forces is not None:
            save_geometries(
                xyz_path,
                inputs,
                self.charges, 
                energy=np.array(energies[fidelity]), 
                forces=forces[fidelity]
            )
        else:
            save_geometries(
                xyz_path, 
                inputs, 
                self.charges, 
                energy=energies[fidelity]
            )

        # conver extended xyz to dataset
        p = subprocess.Popen(
            f'python {self.data_conversion_script_path} {xyz_path}',
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            shell=True,
            text=True
        )
        stdout_data, stderr_data = p.communicate("\n".join(["", "", "", "", "", ""]) + "\n")
        if p.returncode != 0:
            print(stderr_data)
        else:
            print(stderr_data)
            print(stdout_data)


        npz_name = f'train_dataset_{fidelity}.npz'
        npz_path = join(self.model_dir, npz_name)
        shutil.copy(npz_name, npz_path)
        os.remove(npz_name)

        # train model
        dataset = np.load(npz_path)

        n_train = len(inputs)

        gdml_train = GDMLTrain(**self.train_kwargs)
        task = gdml_train.create_task(
            dataset, 
            n_train,
            valid_dataset=dataset,
            n_valid=0,
            **self.task_kwargs
        )

        model = gdml_train.train(task)
        np.savez(join(self.model_dir, f'model_{fidelity}.npz'), **model)

    def predict_single_fidelity_(self, inputs, fidelity=None):

        fidelity = fidelity or self.fidelity

        model = np.load(join(self.model_dir, f'model_{fidelity}.npz'))
        gdml = GDMLPredict(model)
        e, f = gdml.predict(inputs)

        return torch.tensor(e), torch.tensor(f).reshape(inputs.shape[0], int(inputs.shape[1]/3), 3)
    

class SeparateFidelitiesSGDML(SingleFidelitySGDML):
    def __init__(self, *args, low_fidelity=None, **kwargs):
        self.low_fidelity = low_fidelity
        super(SeparateFidelitiesSGDML, self).__init__(*args, **kwargs)

    def fit(self, inputs, energies, forces=None):
        self.fit_single_fidelity_(inputs, energies, forces=forces)
        self.fit_single_fidelity_(inputs, energies, forces=forces, fidelity=self.low_fidelity)

    def uncertainty(self, inputs, low_fidelity_energies=None, low_fidelity_forces=None):
        
        preds_e, preds_f = self.predict_single_fidelity_(inputs, fidelity=self.low_fidelity) 

        u_e = preds_e - low_fidelity_energies
        u_f = preds_f - low_fidelity_forces

        return u_e, u_f


class DeltaSGDML(SingleFidelitySGDML):
    def __init__(self, *args, low_fidelity=None, **kwargs):
        self.low_fidelity = low_fidelity
        super(DeltaSGDML, self).__init__(*args, **kwargs)

    def fit(self, inputs, energies, forces=None):
        energies_new = {
            self.fidelity: energies[self.fidelity] - energies[self.low_fidelity],
            self.low_fidelity: energies[self.low_fidelity]
        }
        forces_new = {
            self.fidelity: forces[self.fidelity] - forces[self.low_fidelity],
            self.low_fidelity: forces[self.low_fidelity]
        }
        self.fit_single_fidelity_(inputs, energies_new, forces=forces_new)  # train on delta
        self.fit_single_fidelity_(
            inputs, energies_new, forces=forces_new, fidelity=self.low_fidelity
        )  # train on low fidelity

    def predict(self, inputs, low_fidelity_energies=None, low_fidelity_forces=None):
        preds_e, preds_f = self.predict_single_fidelity_(inputs) 
        
        preds_e += low_fidelity_energies
        preds_f += low_fidelity_forces
        return preds_e, preds_f
    
    def uncertainty(self, inputs, low_fidelity_energies=None, low_fidelity_forces=None):
        
        preds_e, preds_f = self.predict_single_fidelity_(inputs, fidelity=self.low_fidelity)

        u_e = preds_e - low_fidelity_energies
        u_f = preds_f - low_fidelity_forces

        return u_e, u_f


def save_geometries(path, geometries, charges, **properties):
    ase_trajectory = geometries_to_ase_atoms(geometries, charges, **properties)
    write(path, ase_trajectory)


def create_atoms_object(geometry: torch.Tensor, charges: torch.Tensor, properties: dict) -> Atoms:
    """Creates an ASE Atoms object from geometry, charges, and other properties."""
    n_atoms = int(len(geometry) / 3)

    atoms = Atoms(positions=geometry.reshape((n_atoms, 3)), numbers=charges)

    if properties:
        props = {k: v for k, v in properties.items()}
        atoms.calc = SinglePointCalculator(atoms, **props)

    return atoms


def geometries_to_ase_atoms(
        geometries: torch.Tensor,
        charges: torch.Tensor,
        **properties: Dict[str, np.ndarray]
) -> List[Atoms]:
    """Converts tensor of geometries to list with ASE atoms, handling optional properties like energies and forces."""
    ase_atoms_l = []
    for i, geometry in enumerate(geometries):
        props_for_atoms = {prop: properties[prop][i] for prop in properties}
        atoms = create_atoms_object(geometry, charges, props_for_atoms)
        ase_atoms_l.append(atoms)
    return ase_atoms_l

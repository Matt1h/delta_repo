from typing import Dict
import numpy as np


UNITS_SCALE = {
    'eV': 1,
    'kcal/mol': 23.0621,
}


def no_representation():
    return None


def full():
    return None


def first_n_samples(n_samples):
    return np.arange(n_samples)


def load_scalar_properties_npz(npz_path: str, fidelity_keys: Dict, unit='eV'):

    properties = {}
    for fidelity_name, npz_key in fidelity_keys.items():
        property_values = load_npz(npz_path, key=npz_key)/UNITS_SCALE[unit]
        if property_values.shape[-1] == 1:
            property_values = property_values.squeeze()

        properties[fidelity_name] = (np.arange(len(property_values)), property_values)

    return properties



def load_npz(npz_path: str, key=None):
    data = np.load(npz_path)
    return data[key]

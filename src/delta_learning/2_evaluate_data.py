from os.path import  join
import socket
import logging
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt

from dvclive import Live

log = logging.getLogger(__name__)


def main():
    log.info("Running on host: " + str(socket.gethostname()))

    cfg = OmegaConf.load("params.yaml")

    energies = np.load(join('results', 'data', 'new_data', 'energies.npz'))

    with Live('dvclive/evaluate_data') as live:
        plot_first_n_energies(energies, 100, cfg.dataset, live)
        plot_first_n_energies(energies, 1000, cfg.dataset, live)

        for fidelity_name, values in energies.items():
            live.log_metric(f'MAD_{fidelity_name}', abs(values[1] - values[1].mean()).mean(), plot=False)


def plot_first_n_energies(energies, n_energies, cfg_dataset, live):
    n_fidelities = len(energies)
    fig, axes = plt.subplots(n_fidelities + 1, 1, sharex=True, figsize=(n_fidelities * 3, 5))

    for i, (fidelity_name, values) in enumerate(energies.items()):
        if len(values[1]) >= n_energies:
            config_inds = np.arange(n_energies)
        else:
            config_inds = np.arange(len(values[1]))
        axes[i].scatter(config_inds, values[1][:n_energies])
        axes[i].set_title(fidelity_name)

    energy_diff = (
            energies[cfg_dataset.high_fidelity][1][:n_energies] - energies[cfg_dataset.low_fidelity][1][:n_energies]
    )
    axes[-1].scatter(config_inds, energy_diff)
    axes[-1].set_title('Difference')

    plt.tight_layout()
    live.log_image(f'energy_fidelities_first{len(config_inds)}.png', fig)


if __name__ == '__main__':
    main()

from os.path import  join
import logging
import socket
import torch
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from omegaconf import OmegaConf
from dvclive import Live


log = logging.getLogger(__name__)


def main():
    log.info("Running on host: " + str(socket.gethostname()))

    cfg = OmegaConf.load("params.yaml")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)

    with Live('dvclive/evaluate') as live:
        # energies
        refs_e_test = torch.load(join('results', 'prediction', 'refs_e_test'), weights_only=True)
        preds_e_test = torch.load(join('results', 'prediction','preds_e_test'), weights_only=True)
        live.log_metric('MAE energy', float(abs(refs_e_test - preds_e_test).mean()), plot=False)
        log_erd_energy(live, cfg)

        # forces
        if cfg.model.use_forces:
            refs_f_test = torch.load(join('results', 'prediction', 'refs_f_test'), weights_only=True)
            preds_f_test = torch.load(join('results', 'prediction','preds_f_test'), weights_only=True)
            live.log_metric('MAE forces', float(abs(refs_f_test - preds_f_test).mean()), plot=False)
            log_erd_forces(live, cfg)


def log_erd_energy(live, cfg):
    preds_e_al = torch.load(join('results', 'prediction','preds_e_al'), weights_only=True)
    refs_e_al = torch.load(join('results', 'prediction', 'refs_e_al'), weights_only=True)
    errors_e_al = preds_e_al - refs_e_al
    uncertainties_e_al = torch.load(join('results', 'prediction', 'uncertainties_e_al'), weights_only=True)

    fig = get_erd(
        abs(uncertainties_e_al), 
        abs(errors_e_al), 
        cfg, 
        title='ERD energies'
    )
    live.log_image(f'erd_energy.png', fig)


def log_erd_forces(live, cfg):

    # plot correlation
    preds_f_al = torch.load(join('results', 'prediction','preds_f_al'), weights_only=True)
    refs_f_al = torch.load(join('results', 'prediction', 'refs_f_al'), weights_only=True)
    errors_f_al = preds_f_al - refs_f_al
    uncertainties_f_al = torch.load(join('results', 'prediction', 'uncertainties_f_al'), weights_only=True)

    fig = get_erd(
        abs(uncertainties_f_al).flatten(), 
        abs(errors_f_al).flatten(), 
        cfg, 
        title='ERD forces separate'
    )
    live.log_image(f'erd_forces_separate.png', fig)

    fig = get_erd(
        abs(uncertainties_f_al).mean(dim=2).flatten(), 
        abs(errors_f_al).mean(dim=2).flatten(), 
        cfg, 
        title='ERD forces atomistc'
    )
    live.log_image(f'erd_forces_atomistic.png', fig)

    fig = get_erd(
        abs(uncertainties_f_al).mean(dim=2).mean(dim=1), 
        abs(errors_f_al).mean(dim=2).mean(dim=1), 
        cfg,
        title='ERD forces averaged'
        )
    live.log_image(f'erd_forces.png', fig)


def get_erd(uncertainties, errors, cfg, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax = instantiate(cfg['s6_evaluate']['plot_function'])(
        ax,
        uncertainties.detach()*1000,
        errors.detach()*1000,
        **cfg['s6_evaluate']['plot_function_kwargs']
    )

    ax.set_xlabel('Uncertainty in meV', fontsize=14)
    ax.set_ylabel('Error in meV', fontsize=14)

    ax.set_title(f'{title} {cfg.dataset.name} {cfg.dataset.molecule_name}', fontsize=16)

    plt.tight_layout()
    return fig 


if __name__ == '__main__':
    main()

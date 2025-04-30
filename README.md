 # Short description  
[DVC](https://dvc.org/) repository containing a pipeline to run MLIP training on two different energy and force fidelities, as well as on their difference.  
Configurations are organized using [Hydra](https://hydra.cc/docs/intro/). You can choose between an [sGDML](https://sgdml.org/) model or a GPR model based on [GPyTorch](https://gpytorch.ai/).


# Installation
Clone the repository and run
```bash
pip install -e .
```
inside it. Note that the [GPR_MLIP](https://github.com/SM4DA/GPR_MLIP_uncertainty_evaluation) package has to be installed.

# Prepare data for experiments
Before running experiments, the data has to be stored appropriately. To run the experiments with the datasets used in the paper, create a directory named `datasets` in the same directory as this repository. Inside the `datasets` directory, include the following:

- For the **rMD17** dataset:
  - Download and extract the `.zip` file from [rMD17 on Figshare](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038).
  - Place the extracted `rmd17` directory directly inside the `datasets` directory, i.e., `datasets/rmd17`.

- For the **WS22** dataset:
  - Download the `.npz` files for the individual molecules from [WS22 on Zenodo](https://zenodo.org/records/7032334).
  - Create a directory `WS22` inside the `datasets` directory and place all `.npz` files there, i.e., `datasets/WS22/*.npz`.

If your datasets are stored in different locations, or if you want to use other datasets, you can specify the dataset paths by overriding the corresponding Hydra configurations.

# Run pipeline
Inside the repository run
```bash
dvc exp run
```
## Stages 
The pipeline consists of six stages:

1. `calc_new_data`: Calculate energies and forces with XTB as the low-fidelity method.
2. `evaluate_data`: Calculate the MAD (mean absolute deviation) for both fidelities and create plots showing the differences between the energy values of the two fidelities.
3. `prepare_data`: Randomly shuffle and split the data into train, test, and active learning sets.
4. `train`: Train the model on the training data.
5. `predict`: Predict energies and uncertainties for the test and active learning sets.
6. `evaluate`: Generate extended reliability diagrams on the active learning set and calculate errors on the test set.

## Models

With the `model` configuration, you can choose between different models.
You can either select an [sGDML](https://sgdml.org/) model or a GPR model based on [GPyTorch](https://gpytorch.ai/), which can be:

- trained on the single high-fidelity data,
- trained separately on both fidelities, or
- trained on the difference between the fidelities.

It is also possible to use a model that predicts the high-fidelity energy by adding the difference between the means of the training energies of the two fidelities as a bias to the low-fidelity energy value.  
Depending on the selected option, different uncertainties are calculated.


### GPR models based on GPyTorch
The GPR models based on GPyTorch are only trained on energies.  
The hyperparameters are loaded from the `gpr_models` directory; currently, the models are not trained within the repository.  
Three different model options are available:

```bash
dvc exp run -S model=single_fidelity_gpr
```
Here, the GPR standard deviation is used as the uncertainty.

```bash
dvc exp run -S model=delta_gpr
```
Here, the GPR standard deviation is used as the uncertainty as well.

```bash
dvc exp run -S model=separate_fidelities_gpr
```
Here, the uncertainty is calculated as the difference between the prediction of the low-fidelity energy and the actual low-fidelity energy.


### sGDML
sGDML models are trained on the forces but can also predict energies. Uncertainties are returned for energies as well as for force components. 
Three different model options are available:

```bash
dvc exp run -S model=single_fidelity_sgdml
```
Here, random values are returned as the uncertainty.

```bash
dvc exp run -S model=delta_sgdml
```
Here, the uncertainty is calculated as the difference between the prediction of the low-fidelity energy and the actual low-fidelity energy.

```bash
dvc exp run -S model=separate_fidelities_sgdml
```
Here, the uncertainty is calculated as the difference between the prediction of the low-fidelity energy and the actual low-fidelity energy.


### Add delta
```bash
dvc exp run -S model=add_delta
```
Here random values are returned as uncertainty.

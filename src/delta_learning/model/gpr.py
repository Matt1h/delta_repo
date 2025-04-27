import os
from os.path import join
import torch
import copy

from GPR_MLIP.models import GaussianProcessRegression, std_dev


class SingleFidelityGPR:
    def __init__(
            self,
            fidelity,
            model_kwargs=None,
            kernel=None, 
            old_state_dict_path=None, 
            uncertainty_kwargs=None
        ):
        self.model_path = join('results', 'model', 'state_dict.pt')
        self.fidelity = fidelity
        self.model_kwargs = model_kwargs
        self.kernel = kernel
        self.old_state_dict_path = old_state_dict_path
        self.uncertainty_kwargs = uncertainty_kwargs
        
    def fit(self, train_inputs, train_targets):

        torch.save(train_inputs, join('results', 'model', 'train_inputs.pt'))
        torch.save(train_targets[self.fidelity], join('results', 'model', 'train_targets.pt'))

        if self.old_state_dict_path is not None:
            state_dict = torch.load(
                join(os.path.expanduser("~"), 'dev', 'delta_repo', self.old_state_dict_path),
                weights_only=False
            )
            torch.save(state_dict, self.model_path)
            return

        # kernel = self.kernel(ard_num_dims=train_inputs.shape[1])
        # model = GaussianProcessRegression(kernel=kernel, **self.model_kwargs)
        # model.fit(train_inputs, train_targets[self.fidelity])
        # model.update_hyper_inits()
        # torch.save(model.state_dict(), self.model_path)

    def predict(self, inputs, forces=False):
        model = self.load_model()
        return model.predict(inputs)
        

    def uncertainty(self, inputs, forces=False):
        model = self.load_model()
        return std_dev(model, inputs, **self.uncertainty_kwargs)

    def load_model(self):
        train_inputs = torch.load(join('results', 'model', 'train_inputs.pt'), weights_only=True)
        train_targets =torch.load(join('results', 'model', 'train_targets.pt'), weights_only=True)

        kernel = self.kernel(ard_num_dims=train_inputs.shape[1])
        model = GaussianProcessRegression(kernel=kernel, **self.model_kwargs)
        state_dict = torch.load(join('results', 'model', 'state_dict.pt'), weights_only=False)
        model.load_state_dict(state_dict)
        model.update_hyper_inits()

        model.fit_with_fixed_hypers(train_inputs, train_targets)
        return model


class DeltaGPR(SingleFidelityGPR):
    def __init__(self, *args, low_fidelity=None, **kwargs):
        self.low_fidelity = low_fidelity
        super(DeltaGPR, self).__init__(*args, **kwargs)

    def fit(self, train_inputs, train_targets):

        train_targets_new = {f'{self.fidelity}': train_targets[self.fidelity] - train_targets[self.low_fidelity]}  # train on diff
        super().fit(train_inputs, train_targets_new)

    def predict(self, inputs, low_fidelity_targets=None):
        return super().predict(inputs) + low_fidelity_targets
    
    def uncertainty(self, inputs, forces=False):
        model = self.load_model()
        return std_dev(model, inputs, **self.uncertainty_kwargs)



class SeparateFidelitiesGPR:
    """Model with submodels for different fidelities"""
    def __init__(
            self,
            fidelity, 
            low_fidelity=None, 
            model_kwargs=None, 
            kernel=None, 
            old_state_dict_path=None, 
            uncertainty_kwargs=None
        ):
        self.model_path = join('results', 'model', 'state_dict.pt')
        self.fidelity = fidelity
        self.low_fidelity = low_fidelity
        self.model_kwargs = model_kwargs
        self.kernel = kernel
        self.old_state_dict_path = old_state_dict_path
        self.uncertainty_kwargs = uncertainty_kwargs

    def fit(self, train_inputs, train_targets):

        torch.save(train_inputs, join('results', 'model', 'train_inputs.pt'))
        torch.save(train_targets, join('results', 'model', 'train_targets.pt'))
        
        if self.old_state_dict_path is not None:
            state_dict = torch.load(
                join(os.path.expanduser("~"), 'dev', 'delta_repo', self.old_state_dict_path), 
                weights_only=False
            )
            torch.save(state_dict, self.model_path)
            return

        # for fidelity in [self.fidelity, self.low_fidelity]:
        #     kernel = self.kernel(ard_num_dims=train_inputs.shape[1])
        #     model = GaussianProcessRegression(kernel=kernel, **self.model_kwargs)
        #     model.fit(train_inputs, train_targets[fidelity])
        #     model.update_hyper_inits()
        #     torch.save(model.state_dict(), self.model_path)

    def predict(self, inputs, forces=False):
        models = self.load_models()
        return models[self.fidelity].predict(inputs)

    def uncertainty(self, inputs, low_fidelity_targets=None):
        models = self.load_models()
        return abs(models[self.low_fidelity].predict(inputs) - low_fidelity_targets)
    

    def load_models(self):
        models = {}
        for fidelity in [self.fidelity, self.low_fidelity]:
            train_inputs = torch.load(join('results', 'model', 'train_inputs.pt'), weights_only=True)
            train_targets =torch.load(join('results', 'model', 'train_targets.pt'), weights_only=True)[fidelity]

            kernel = self.kernel(ard_num_dims=train_inputs.shape[1])
            model = GaussianProcessRegression(kernel=kernel, **self.model_kwargs)
            state_dict = torch.load(join('results', 'model', 'state_dict.pt'), weights_only=False)
            model.load_state_dict(state_dict)
            model.update_hyper_inits()
            model.fit_with_fixed_hypers(train_inputs, train_targets)

            models[fidelity] = copy.deepcopy(model)

        return models

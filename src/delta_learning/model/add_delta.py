from os.path import join
import torch


class AddDelta:
    def __init__(self, fidelity, low_fidelity=None):
        self.model_path = join('results', 'model', 'state_dict.pt')
        self.fidelity = fidelity
        self.low_fidelity = low_fidelity

    def fit(self, train_inputs, train_targets):
        delta = (train_targets[self.fidelity] - train_targets[self.low_fidelity]).mean()
        torch.save(delta, self.model_path)

    def predict(self, inputs, low_fidelity_targets=None):
        delta = torch.load(self.model_path, weights_only=True)
        return delta + low_fidelity_targets
    
    def uncertainty(self, inputs, forces=False):
        return torch.rand(len(inputs))
import torch
from matplotlib import pyplot as plt
import numpy as np

dataset_name  = "small_island_manual_200"
train_easy = torch.from_numpy(np.load(dataset_name +"_train.npy"))
valid_easy = torch.from_numpy(np.load(dataset_name +"_valid.npy"))

dataset_name  = "small_island_hard_200"
train_medium = torch.from_numpy(np.load(dataset_name +"_train.npy"))
valid_medium = torch.from_numpy(np.load(dataset_name +"_valid.npy"))

dataset_name  = "small_island_tough_200"
train_hard = torch.from_numpy(np.load(dataset_name +"_train.npy"))
valid_hard = torch.from_numpy(np.load(dataset_name +"_valid.npy"))



class EnergyPredictor:
    def __init__(self, X, T=1.):
        '''
        X : [trajectory num x state dim x horizon] tensor
        '''
        self.T = T
        self.probs = self._fit_probs(self.phi(X.movedim(0, -1)).movedim(-1, 0))

    def phi(self, data):
        '''
        Converts data to feature space
        - data : [state dim x horizon, ...] tensor
        output: [K, ...]
        '''
        feats = torch.cat( (torch.amax(data[bound_state_inds], dim=1), torch.amin(data[bound_state_inds], dim=1)), dim=0)
        return feats

    def ood_energy(self, data, T=1.0):
        # data : [trajectory num x state dim x horizon] tensor
        ps = self._comp_probs(self.phi(data.movedim(0,-1)).movedim(-1, 0))
        return -T * torch.logsumexp(ps / T, dim=-1)

    def _fit_probs(self, feats):
        '''
        Fits a multivariate normal distribution to the data
        - feats: [trajectory num x feature dim] tensor
        '''
        # data = data.numpy()
        mu = torch.mean(feats, axis=0)
        var = torch.var(feats, axis=0) # adjust for biased estimate
        return [norm(loc=mu[i], scale=var[i]) for i in range(len(mu))]

    def _comp_probs(self, feats):
        '''
         - feats : [trajectory num x feature dim] tensor
        output : [trajectory num x feature dim] tensor of probabilities
        '''
        ps = [torch.tensor(d.pdf(feats[..., i]), dtype=feats.dtype, device=feats.device) for i,d in enumerate(self.probs)]
        return torch.stack(ps, dim=-1)

bound_state_inds = [9,10,11] # indices corresponding to state component to be bounded

fit = EnergyPredictor(train_easy)

train_E = fit.ood_energy(train_easy)
valid_E = fit.ood_energy(valid_easy)
id_E = fit.ood_energy(train_medium) # replace test_set[Y] with median
ood_E = fit.ood_energy(train_hard) # replace test_set[~Y] with hard

args = {'density': True, 'bins' : 50}
plt.hist(train_E, color='b', alpha=0.3, label='easy', **args)
plt.hist(ood_E, color='g', alpha=0.3, label='hard', **args)
plt.hist(id_E, color='r', alpha=0.3, label='medium', **args)
plt.ylabel("Energy")
plt.xlabel("")
plt.legend()
plt.show()
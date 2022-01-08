import torch
import torch.nn as nn
import torch.nn.functional as F
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np

class SpirlPolicy(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        embedding,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.spirl = True
        self.z_dim = embedding


    def get_action(self, obs_np, deterministic=False):
        action, z = self.get_actions(obs_np[None], deterministic=deterministic)
        return action[0, :], z[0,:],{}

    def get_actions(self, obs_np, deterministic=False):
        z = eval_np(self, obs_np, deterministic=deterministic)[0]
        action = eval_np(self.decoder,z)
        return action,z

    def encode(self,obs):
        z_params =  self.encoder(obs)
        z_mu,z_std = z_params[:,:self.z_dim],F.softplus(z_params[:,self.z_dim:])
        return z_mu,z_std

    def reparameterize(self,z_mean, z_var):
        pi_z = torch.distributions.normal.Normal(z_mean, z_var)
        return pi_z

    def forward(
        self, obs, reparameterize=True, deterministic=False, return_log_prob=False,
    ):
        z_mu,z_std =  self.encode(obs)
        if deterministic:
            z = z_mu
            z_std = torch.zeros_like(z_std)
            pi_z = self.reparameterize(z,z_std)
        else:
            pi_z = self.reparameterize(z_mu,z_std)
            if reparameterize:
                z = pi_z.rsample()
            else:
                z = pi_z.sample()
        return (z,
                z_mu,
                z_std,
                pi_z)

    def reset(self):
        None

    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
        return n_params.item()

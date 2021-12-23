import torch
import torch.nn as nn
import torch.nn.functional as F
from rlkit.torch import pytorch_util as ptu

class SPiRL_policy(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        embedding,
    ):
        super().__init__()
        self.policy_enc = encoder
        self.prior_decoder = decoder
        self.spirl = True
        self.z_dim = embedding
        
        
    def get_action(self, obs_np, deterministic=False):
        action, z = self.get_actions(obs_np[None], deterministic=deterministic)
        return action[0, :], z[0,:],{}
    
    def get_actions(self, obs_np, deterministic=False):
        z = eval_np(self, obs_np, deterministic=deterministic)[0]
        action = eval_np(self.decoder,z)
        return action,z
    
    def encode(obs):
        z_params =  self.encoder(obs)
        z_mu,z_std = z_params[:,:self.z_dim],F.softplus(z_params[:,self.z_dim:])
        return z_mu,z_std
    
    def reparameterize(self,z_mean, z_var):
        pi_z = torch.distributions.normal.Normal(z_mean, z_var)
        return pi_z
    
    def forward(
        self, obs, reparameterize=True, deterministic=False, return_log_prob=False,
    ):
        z_mu,z_std =  self.encoder(obs)
        if determinstic:
            z = z_mu
        else:
            pi_z = reparameterize(self,z_mu,z_std)
            if reparameterize:
                z = pi_z.rsample()
            else:
                z = pi_z.sample()
        return (z,
                z_mu,
                z_std,
                pi_z)
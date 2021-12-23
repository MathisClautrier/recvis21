import torch
import torch.nn as nn
import torch.nn.functional as F
from rlkit.torch import pytorch_util as ptu


class DecoderSeq(nn.Module):
    def __init__(
        self,
        embedding_size,
        H,
        hidden_size,
    ):
        super().__init__()
        self.dec = nn.LSTM(embedding_size,hidden_size,batch_first=True) #2 as there are two floats per action
        self.projdec = nn.Linear(hidden_size,2)
        self.z_dim = embedding_size
        self.H = H
        self.hidden = hidden_size
        self.activation = nn.Tanh()
        
    def forward(self, z):
        batch_size = z.shape[0]
        isGPU = z.get_device()
        if isGPU == 0:
            X = torch.zeros((batch_size,self.H,2)).cuda()
        else:
            X = torch.zeros((batch_size,self.H,2))
        z = z[:,None,:]
        params=None
        for i in range(self.H):
            if params is None:
                x,params= self.dec(z)
            else:
                x,params = self.dec(z,params)
            x = self.activation(self.projdec(x[:,-1,:]))
            X[:,i,:] = x
        return X
    
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
        return n_params.item()
    
class EncoderSeq(nn.Module):
    def __init__(
        self,
        embedding_size,
        H,
        hidden_size,
    ):
        super().__init__()
        self.enc= nn.LSTM(2,hidden_size,batch_first=True) #2 as there are two floats per action
        self.projenc = nn.Linear(hidden_size,2*embedding_size)
        self.z_dim = embedding_size
        self.H = H
        self.hidden = hidden_size
        
    def encode(self,actions):
        Zseq,_ = self.enc(actions)
        Zseq = self.projenc(Zseq[:,-1,:])
        mu,sig = Zseq[:,:self.z_dim],F.softplus(Zseq[:,self.z_dim:])
        return mu,sig
    
    def reparameterize(self,z_mean, z_var):
        q_z = torch.distributions.normal.Normal(z_mean, z_var)
        p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        return q_z,p_z
    
    def forward(self,actions):
        mu,std = self.encode(actions)
        q_z,p_z = self.reparameterize(mu,std)
        return (mu,std),(q_z,p_z)
    
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
        return n_params.item()
    
class SkillPrior(nn.Module):
    def __init__(
        self,
        sEncoder,
        embedding_size,
        H,
        hidden_size,
    ):
        super().__init__()
        self.stateEncoder = sEncoder
        self.actionsEncoder = EncoderSeq(embedding_size,H,hidden_size)
        self.actionsDecoder = DecoderSeq(embedding_size,H,hidden_size)
        self.z_dim = embedding_size
        self.H = H
        self.hidden = hidden_size


    def encode(self, states):
        Zstate = self.stateEncoder(states)
        mustate,sigstate = Zstate[:,:self.z_dim],F.softplus(Zstate[:,self.z_dim:])
        return mu,std
    
        
    def reparameterize(self,z_mean, z_var):
        q_z = torch.distributions.normal.Normal(z_mean, z_var)
        p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        return q_z,p_z
    
    def obtain_q_z(self,actions):
        _,dist = self.actionsEncoder(actions)
        q_z = dist[0]
        return q_z
    
    def obtain_pa_z(self,states):
        zs_mean,zs_var = self.encode(states)
        pa_z,_ = self.reparameterize(zs_mean,zs_var)
        return pa_z
               
    def forward(self, states,actions): 
        zs_mean,zs_var = self.encode(states)
        pa_z,_ = self.reparameterize(zs_mean,zs_var)
        (z_mean,z_var),(q_z,p_z) = self.actionsEncoder(actions)
        z = q_z.rsample()
        x_ = self.decode(z)
        
        return (z_mean, z_var,zs_mean,zs_var), (q_z, p_z,pa_z), z, x_
    
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
        return n_params.item()
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
        z = z[:,None,:]
        z = torch.repeat_interleave(z,self.H,1)
        X,_ = self.dec(z)
        X = X.reshape(batch_size*self.H,self.hidden)
        X = self.activation(self.projdec(X))
        X = X.reshape(batch_size,self.H,2)
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

class MlpSkillEncoder(nn.Module):
    def __init__(
        self,
        n_inputs,
        z_dim,
    ):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(n_inputs,256),
          nn.ELU(),
          nn.Linear(256,128),
          nn.ELU(),
          nn.Linear(128,64),
          nn.ELU(),
          nn.Linear(64,32),
          nn.ELU(),
          nn.Linear(32,2*z_dim),
        )
        self.z_dim = z_dim

    def forward(self,inputs):
        h = self.layers(inputs)
        return h

    def obtain_pa_z(self,states):
        h = self.forward(states)
        z_mu,z_std = h[:,:self.z_dim],F.softplus(h[:,self.z_dim:])
        pa_z = torch.distributions.normal.Normal(z_mu, z_std)
        return pa_z


class SkillPrior(nn.Module):
    def __init__(
        self,
        sEncoder,
        embedding_size,
        H,
        hidden_size,
    ):
        super().__init__()
        self.statesEncoder = sEncoder
        self.actionsEncoder = EncoderSeq(embedding_size,H,hidden_size)
        self.actionsDecoder = DecoderSeq(embedding_size,H,hidden_size)
        self.z_dim = embedding_size
        self.H = H
        self.hidden = hidden_size


    def encode(self, states):
        Zstate = self.statesEncoder(states)
        mu,std = Zstate[:,:self.z_dim],F.softplus(Zstate[:,self.z_dim:])
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

    def forward_state(self,states):
        zs_mean,zs_var = self.encode(states)
        pa_z,qa_z = self.reparameterize(zs_mean,zs_var)
        z = pa_z.rsample()
        x_ = self.actionsDecoder(z)
        return (zs_mean,zs_var),(pa_z,qa_z),z,x_

    def forward_actions(self,actions):
        (mu,std),(q_z,p_z) = self.actionsEncoder(actions)
        z = q_z.rsample()
        x_ = self.actionsDecoder(z)
        return (mu,std),(q_z,p_z),z,x_

    def forward(self, states,actions):
        zs_mean,zs_var = self.encode(states)
        pa_z,_ = self.reparameterize(zs_mean,zs_var)
        (z_mean,z_var),(q_z,p_z) = self.actionsEncoder(actions)
        z = q_z.rsample()
        x_ = self.actionsDecoder(z)

        return (z_mean, z_var,zs_mean,zs_var), (q_z, p_z,pa_z), z, x_

    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
        return n_params.item()

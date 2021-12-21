import mpenv.envs
import gym
import numpy as np
from tqdm import tqdm
import random
import click
from nmp.launcher.utils import *
from nmp.model.pointnet import PointNetEncoder
import torch
from nmp.model.skill import SkillPrior
import argparse
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser(description='prior_training')
parser.add_argument('--env-name', type=str, default='Maze-Medium-v0', help='environment ID')
parser.add_argument('--simple-like',  action='store_true', help='Initiate the environment with shortest path similar to the possible ones in SimpleMaze')
parser.add_argument('--seed', type = int, default = 0, help = 'random seed')
parser.add_argument('--embedding-dim', type = int, default = 10, help = 'embedding dimension')
parser.add_argument('--H', type = int, default = 10, help = 'trajectory size')
parser.add_argument('--data-dir', type = str, default = 'not_provided',help='direction to the data')
parser.add_argument('--archi',type = str, default = 'pointnet', help = 'backbone of the state encoder')
parser.add_argument('--hidden-dim',type = int, default = 256)
parser.add_argument('--hidden-dim-lstm',type = int, default = 128)
parser.add_argument('--n-layers', type = int, default =3)
parser.add_argument('--log-dir', type = str, default ='pointnet_3_seed_0')
parser.add_argument('--batch-size',type = int, default = 16)
parser.add_argument('--epochs',type = int, default = 1)
parser.add_argument('--GPU',  action='store_true')
parser.add_argument('--beta', type = float, default = 1e-2)
parser.add_argument('--learning_rate',type = float, default = 1e-3)
parser.add_argument('--beta1', type = float, default = 0.9)
parser.add_argument('--beta2', type = float, default = 0.999)

args = parser.parse_args()

if args.data_dir == 'not_provided':
    print('The directory in which the data is stored must be provided.')
    raise ValueError
    
if args.GPU:
    device = 'cuda'
else:
    device = 'cpu'
    
argsdic = vars(args)

env = gym.make(args.env_name)

argsdic["policy_kwargs"]=dict(hidden_dim=args.hidden_dim, n_layers=args.n_layers)
_, policy_kwargs = get_policy_network(argsdic["archi"], agrsdic["policy_kwargs"], env, "vanilla")

Enc = PointNetEncoder(**policy_kwargs, embedding = args.embedding_dim)

skillP = SkillPrior(Enc,args.embedding_dim,args.H,args.hidden_dim_lstm)
skillP.to(device)
class SkillDataset(torch.utils.data.Dataset):
    def __init__(self, directory,H):
        'Initialization'
        self.files = [f for f in listdir(directory) if isfile(join(directory, f))]
        self.dir = directory
        self.keys = ['sequence','state']
        self.H = H

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.files)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        data = np.load(self.dir+'/'+self.files[index])
        A,S = (data[k] for k in self.keys) 
        A = A.reshape(self.H,2)
        return S,A

dataset = SkillDataset(args.data_dir,args.H)
training = torch.utils.data.DataLoader(training_set, batch_size = args.batch_size,shuffle = True)

optimizer = optim.RAdam(skillP.parameters(), lr=args.lr, betas=[args.beta1,args.beta2])

for epoch in range(args.epochs):
    running_loss=0
    m=0
    for states,actions in training:
        
        optimizer.zero_grad()
        z_mu,z_var,zs_mu,zs_var,q_z,p_z,pa_z,z,actions_ =  skillP.forward(states.to(device),actions.to(device))
        
        loss = torch.square(actions - actions_).sum(axis=1).mean(axis=0).mean()
        
        loss += -args.beta*torch.distributions.kl.kl_divergence(q_z,p_z).sum(axis=1).mean()
        
        with torch.no_grad():
            q_z_no_grad = skillP.obtain_q_z(actions.to(device))
        loss += args.beta*torch.distributions.kil.kil_divergence(q_z_no_grad,pa_z)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        m+=1
        
    print(running_loss/m)










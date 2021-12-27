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
import os
import logging

parser = argparse.ArgumentParser(description='prior_training')
parser.add_argument('--env-name', type=str, default='Maze-Medium-v0', help='environment ID')
parser.add_argument('--seed', type = int, default = 0, help = 'random seed')
parser.add_argument('--embedding-dim', type = int, default = 10, help = 'embedding dimension')
parser.add_argument('--H', type = int, default = 10, help = 'trajectory size')
parser.add_argument('--data-dir', type = str, default = 'not_provided',help='direction to the data must contain a training folder and a validation folder')
parser.add_argument('--archi',type = str, default = 'pointnet', help = 'backbone of the state encoder')
parser.add_argument('--hidden-dim',type = int, default = 256)
parser.add_argument('--hidden-dim-lstm',type = int, default = 128)
parser.add_argument('--n-layers', type = int, default =3)
parser.add_argument('--log-dir', type = str, default ='.')
parser.add_argument('--log-name', type = str, default ='pointnet_medium_seed_0')
parser.add_argument('--batch-size',type = int, default = 16)
parser.add_argument('--epochs',type = int, default = 1)
parser.add_argument('--GPU',  action='store_true')
parser.add_argument('--beta', type = float, default = 1e-2)
parser.add_argument('--lr',type = float, default = 1e-3)
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
_, policy_kwargs = get_policy_network(argsdic["archi"], argsdic["policy_kwargs"], env, "vanilla")

Enc = PointNetEncoder(**policy_kwargs, embedding = args.embedding_dim)

model = SkillPrior(Enc,args.embedding_dim,args.H,args.hidden_dim_lstm)
model.to(device)

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
        return torch.Tensor(S),torch.Tensor(A)

def log(path, file):
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(message)s"
    file_logging_format = "%(message)s"
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    handler = logging.FileHandler(log_file)

    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger

logger = log(args.log_dir,args.log_name+'.csv')
datasetT = SkillDataset(args.data_dir+'/training',args.H)
trainingLoader = torch.utils.data.DataLoader(datasetT, batch_size = args.batch_size,shuffle = True)
datasetV = SkillDataset(args.data_dir+'/validation',args.H)
validationLoader = torch.utils.data.DataLoader(datasetV, batch_size = 100)

optimizer= torch.optim.Adam(list(model.statesEncoder.parameters())+list(model.actionsDecoder.parameters()), lr=args.lr, betas=[args.beta1,args.beta2])

for epoch in tqdm(range(args.epochs)):
    tL1,tL2,m=0,0,0
    model.train()
    for states,actions in trainingLoader:
        actions=actions.to(device)
        (zs_mu,zs_var),(qa_z,pa_z),z,actions_ = model.forward_state(states.to(device))

        loss = torch.square(actions - actions_).sum(axis=1).mean(axis=0).mean()
        tL1+=loss.item()
        loss += args.beta*torch.distributions.kl.kl_divergence(qa_z,pa_z).sum(axis=1).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tL1+=MSE.item()
        tL2+=loss.item()
        m+=1
    model.eval()
    
    vL1,vL2,k=0,0,0
    for states,actions in validationLoader:
        actions=actions.to(device)
        (zs_mu,zs_var),(qa_z,pa_z),z,actions_ = model.forward_state(states.to(device))
        
        loss = torch.square(actions - actions_).sum(axis=1).mean(axis=0).mean()
        vL1+=loss.item()
        loss += args.beta*torch.distributions.kl.kl_divergence(qa_z,pa_z).sum(axis=1).mean()

        vL2+=loss.item()
        k+=1
    logger.info(str(epoch)+','+str(tL1/m)+','+str(tL2/m)+','+str(vL1/k)+','+str(vL2/k))

torch.save(model.statesEncoder.state_dict(), args.log_dir+'/'+args.log_name+'_prior.pth')
torch.save(model.actionsDecoder.state_dict(), args.log_dir+'/'+args.log_name+'_dec.pth')
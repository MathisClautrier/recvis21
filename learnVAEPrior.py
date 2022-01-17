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
parser.add_argument('--warmup', type =int, default = 50, help = 'number of warmup epochs')
parser.add_argument('--resume-model', type =str, default = 'not')

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

input_size = env.observation_space.spaces["observation"].low.size
Enc = MlpSkillEncoder(input_size+2,args.embedding_dim)
print(Enc)
model = SkillPrior(Enc,args.embedding_dim,args.H,args.hidden_dim_lstm)
if args.resume_model != 'not':
    old_dec = torch.load(args.resume_model+'_dec.pth', map_location = torch.device(device))
    model.actionsDecoder.load_state_dict(old_dec)
    old_enc = torch.load(args.resume_model+'_enc.pth', map_location = torch.device(device))
    model.actionsEncoder.load_state_dict(old_enc)
    print('resumed model loaded')
model.to(device)



class SkillDataset(torch.utils.data.Dataset):
    def __init__(self, directory,H):
        'Initialization'
        self.files = [f for f in listdir(directory) if isfile(join(directory, f))]
        self.dir = directory
        self.keys = ['sequence']
        self.H = H

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.files)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        data = np.load(self.dir+'/'+self.files[index])
        A, = (data[k] for k in self.keys)
        A = A.reshape(self.H,2)
        return torch.Tensor(A)

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


optimizer= torch.optim.Adam(list(model.actionsEncoder.parameters())+list(model.actionsDecoder.parameters()), lr=args.lr, betas=[args.beta1,args.beta2])
def linearWarmUP(epochs):
    if epochs >= args.warmup:
        return 1
    else:
        return epochs/args.warmup

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = linearWarmUP,verbose = True)
for epoch in tqdm(range(args.epochs+1)):
    tL1,tL2,m=0,0,0
    model.train()


    for actions in trainingLoader:
        actions=actions.to(device)
        (z_mu,z_var),(q_z,p_z),z,actions_ =  model.forward_actions(actions)


        loss = torch.nn.MSELoss(reduction='none')(actions, actions_).sum(axis=-1).sum(axis=-1).mean()
        tL1+=loss.item()
        loss += args.beta*torch.distributions.kl.kl_divergence(q_z,p_z).sum(axis=1).mean()



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tL2+=loss.item()
        m+=1
    model.eval()
    scheduler.step()

    vL1,vL2,k=0,0,0
    for actions in validationLoader:
        with torch.no_grad():
            actions=actions.to(device)
            (z_mu,z_var),(q_z,p_z),z,actions_ =  model.forward_actions(actions)


            loss = torch.nn.MSELoss(reduction='none')(actions, actions_).sum(axis=-1).sum(axis=-1).mean()
            vL1 += loss.item()
            loss += args.beta*torch.distributions.kl.kl_divergence(q_z,p_z).sum(axis=1).mean()

            vL2+=loss.item()
            k+=1
    logger.info(str(epoch)+','+str(tL1/m)+','+str(tL2/m)+','+str(vL1/k)+','+str(vL2/k))


torch.save(model.actionsEncoder.state_dict(), args.log_dir+'/'+args.log_name+'_enc.pth')
torch.save(model.actionsDecoder.state_dict(), args.log_dir+'/'+args.log_name+'_dec.pth')

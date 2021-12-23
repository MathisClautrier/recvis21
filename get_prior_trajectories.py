import mpenv.envs
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from nmp.launcher.utils import *
from nmp.model.pointnet import PointNetEncoder
import torch
from nmp.model.skill import SkillPrior
import argparse

parser = argparse.ArgumentParser(description='prior_training')
parser.add_argument('--env-name', type=str, default='Maze-Medium-v0', help='environment ID')
parser.add_argument('--seed', type = int, default = 0, help = 'random seed')
parser.add_argument('--H', type = int, default = 10, help = 'trajectory size')
parser.add_argument('--model-dir', type = str, default = 'not_provided',help='direction to the saved model, if not provided use an oracle')
parser.add_argument('--noisy', action = 'store_true', help ='oracle mode')
parser.add_argument('--log-dir', type = str, default ='.')
parser.add_argument('--epochs',type = int, default = 1)
parser.add_argument('--GPU',  action='store_true')

args = parser.parse_args()

kwargs=dict(hidden_dim=args.hidden_dim, n_layers=args.n_layers)
env = gym.make(args.env_name)
if args.GPU:
    device = 'cuda'
else:
    device = 'cpu'
    
if args.model_dir != 'not_provided':
    load_models = torch.load(args.model_dir,map_location=torch.device('cpu'))
    policy=load_params['evaluation/policy']
    policy.stochastic_policy.to(device)
if  not 'Simple' in args.env_name:
    env.turn_simple_like()

random.seed(args.seed)
np.random.seed(args.seed)
count=0
H = args.H
while count < 10000:
    state = env.reset()
    done = False
    states = []
    actions=[]
    if args.model_dir != 'not_provided':
        t = 0
        while not done and t <1000:
            action,_=policy.get_action(state['observation'])
            states.append(state)
            actions.append(action)
            state,reward,done,info = env.step(action)
            done = done[0]
            t+=1
        if done:
            N = len(actions)-H
            for i in range(N):
                S = states[i]['observation']
                AQ = states[i]['achieved_q']
                DQ = states[i]['desired_q']
                RG = states[i]['representation_goal']
                A = np.concatenate(actions[i:i+H])
                np.savez(args.log_dir+'/training/'+str(count),state =S, sequence = A, achieved_q = AQ, desired_q = DQ, representation_goal = RG)
                count+=1
                if count ==10000:
                    break
    else:
        env.shortest_path()
        while not done:
            action = env.one_step_oracle(noisy=args.noisy)
            states.append(state)
            actions.append(action)
            state,reward,done,info = env.step(action)
            done = done[0]
        N = len(actions)-H
        for i in range(N):
            S = states[i]['observation']
            AQ = states[i]['achieved_q']
            DQ = states[i]['desired_q']
            RG = states[i]['representation_goal']
            A = np.concatenate(actions[i:i+H])
            np.savez(args.log_dir+'/training/'+str(count),state =S, sequence = A, achieved_q = AQ, desired_q = DQ, representation_goal = RG)
            count+=1
            if count == 10000:
                break

count=0                
while count < 1500:
    state = env.reset()
    done = False
    states = []
    actions=[]
    if args.model_dir != 'not_provided':
        t = 0
        while not done and t <1000:
            action,_=policy.get_action(state['observation'])
            states.append(state)
            actions.append(action)
            state,reward,done,info = env.step(action)
            done = done[0]
            t+=1
        if done:
            N = len(actions)-H
            for i in range(N):
                S = states[i]['observation']
                AQ = states[i]['achieved_q']
                DQ = states[i]['desired_q']
                RG = states[i]['representation_goal']
                A = np.concatenate(actions[i:i+H])
                np.savez(args.log_dir+'/validation/'+str(count),state =S, sequence = A, achieved_q = AQ, desired_q = DQ, representation_goal = RG)
                count+=1
                if count ==1500:
                    break
    else:
        env.shortest_path()
        while not done:
            action = env.one_step_oracle(noisy=args.noisy)
            states.append(state)
            actions.append(action)
            state,reward,done,info = env.step(action)
            done = done[0]
        N = len(actions)-H
        for i in range(N):
            S = states[i]['observation']
            AQ = states[i]['achieved_q']
            DQ = states[i]['desired_q']
            RG = states[i]['representation_goal']
            A = np.concatenate(actions[i:i+H])
            np.savez(args.log_dir+'/validation/'+str(count),state =S, sequence = A, achieved_q = AQ, desired_q = DQ, representation_goal = RG)
            count+=1
            if count == 1500:
                break
                    
import gym
import torch
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
import mpenv.envs
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.samplers.data_collector import (
    GoalConditionedPathCollector,
    MdpPathCollector,
)
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.spirl.spirl import SPIRLTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from nmp.model.skill import DecoderSeq, MlpSkillEncoder
from nmp.model.pointnet import  PointNetEncoder
from nmp.policy.spirl import SpirlPolicy

from nmp.launcher import utils
import copy
from nmp.launcher.sac import(get_replay_buffer,
                             get_path_collector,
                            )


def get_networks(variant, expl_env):
    """
    Define Q networks and policy network
    """


    qf_kwargs = variant["qf_kwargs"]
    dir_models = variant["dir_models"]

    if variant['MLP']:
        input_size = expl_env.observation_space.spaces["observation"].low.size
        prior_skill = MlpSkillEncoder(input_size+2,variant["embedding"])
    else:
        _, policy_kwargs = get_policy_network(variant["archi"], variant["policy_kwargs"], expl_env, "vanilla")
        prior_skill = PointNetEncoder(**policy_kwargs, embedding = variant["embedding"])
    prior_skill.load_state_dict(torch.load(dir_models+'_prior.pth'))

    copy_prior = copy.deepcopy(prior_skill)

    decoder = DecoderSeq(variant["embedding"],variant["h"],variant["hidden_dim_lstm"])
    decoder.load_state_dict(torch.load(dir_models+'_dec.pth'))

    policy = SpirlPolicy(copy_prior,decoder,variant["embedding"])

    shared_base = None

    qf_class, qf_kwargs = utils.get_q_network(variant["archi"], qf_kwargs, expl_env,embedding=variant['embedding'])

    qf1 = qf_class(**qf_kwargs)
    qf2 = qf_class(**qf_kwargs)
    target_qf1 = qf_class(**qf_kwargs)
    target_qf2 = qf_class(**qf_kwargs)
    print("Policy:")
    print(policy)

    nets = [qf1, qf2, target_qf1, target_qf2, policy, prior_skill, shared_base]
    print(f"Q function num parameters: {qf1.num_params()}")
    print(f"Policy num parameters: {policy.num_params()}")

    return nets


def spirl(variant):
    expl_env = gym.make(variant["env_name"])
    eval_env = gym.make(variant["env_name"])
    expl_env.seed(variant["seed"])
    eval_env.set_eval()

    mode = variant["mode"]
    archi = variant["archi"]
    if mode == "her":
        variant["her"] = dict(
            observation_key="observation",
            desired_goal_key="desired_goal",
            achieved_goal_key="achieved_goal",
            representation_goal_key="representation_goal",
        )

    replay_buffer = get_replay_buffer(variant, expl_env)
    qf1, qf2, target_qf1, target_qf2, policy, prior_skill,shared_base = get_networks(
        variant, expl_env
    )
    expl_policy = policy
    eval_policy = MakeDeterministic(policy)

    expl_path_collector, eval_path_collector = get_path_collector(
        variant, expl_env, eval_env, expl_policy, eval_policy
    )

    mode = variant["mode"]
    trainer = SPIRLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        prior_skill=prior_skill,
        **variant["trainer_kwargs"],
    )
    if mode == "her":
        trainer = HERTrainer(trainer)
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"],
    )

    algorithm.to(ptu.device)
    algorithm.train()

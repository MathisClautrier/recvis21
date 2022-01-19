# Solving Hard Exploration Problems with RL using Prior Skills

 This repo is related to my final project for the MVA course "Object recognition and computer vision" and is based on the following paper:

Robin Strudel, Ricardo Garcia, Justin Carpentier, Jean-Paul Laumond, Ivan Laptev, Cordelia Schmid\
CoRL 2020

- [Project Page](https://www.di.ens.fr/willow/research/nmp_repr/)
- [Paper](https://arxiv.org/abs/2008.11174)

The purpose of this project was to improve perfomances obtain on difficult maze problems using prior skills as described in:

Karl Pertsch,  Youngwoon Lee, Joseph Lim\
CoRL 2020

- [Project Page](https://clvrai.github.io/spirl/)
- [Paper](https://arxiv.org/abs/2010.11944)

I coded everything based on the original version of this repository. In particular, the PointNetEncoder class is a modification of the PointNet class (almost a copy-paste), as well as the spirl.py launcher and trainSpirl.py. For the SPiRL implementation I modified the SAC implementation (the rlkit one from the original repo) after modifying the samplers and data handlers to replace the actions with embeddings. The rest was coded from scratch.

### Table of Content

- [Setup](#setup)
- [Training](#train)
- [Logging](#logging)
- [Cite](#cite)


## Setup

Download the code
```
git clone https://github.com/MathisClautrier/recvis21
```

To create a new conda environment containing dependencies
```
conda env create -f environment.yml
conda activate nmprepr
```

To update a conda environment with dependencies
```
conda env update -f environment.yml
```

## Train

### Baseline

With the current version one should be able to run the baseline (i.e the model of the original repo) as originally. To do so you can run the following code to train the baseline on the 3x3 maze for 1500 epochs with trajectories of maximum length 80 and default parameters.
```
python -m nmp.train Maze-Simple-v0 maze_baseline --horizon 8 --seed 0 --epochs 1500
```

### SPiRL

The first step to reproduce project's results is to collect trajectories. To do so one can use the proposed oracle on the simplest maze using
```
python get_prior_trajectories.py Maze-Simple-v0 --log-dir $PATH$ --n-samples 10000
```

Note that the data directory should contain folders "training" and "validation". If you want to explore the use of a policy trained with the baseline you should use the the option ```--model-dir``` and specifiy the related path.

Once the trajectories are collected you need to train a VAE on the action sequences using
```
python learnVAE.py --log-dir $PATH$ --log-name $NAME$ --data-dir $PATHTODATA$
```

Then you need to train your prior skills model by using (given that you want a MLP prior skills model)
```
python learnPriorFromVAE.py --log-dir $PATH$ --log-name $NAME$ --data-dir $PATHTODATA$ --model-dir $PATHTOVAE$ --archi mlp
```

Finally, you can train a SPiRL agent by using
```
python -m nmp.trainSpirl Maze-Simple-v0 spirlMLP --horizon 8 --seed 0 --epochs 1500 --archi mlp --dir-models $PATHTOVAEANDPRIORSKILLS$ --load-prior
```

Note that I proposed to use an horizon of 8 as each action (i.e embedding) yields 10 environment steps.


### Results

Results will be shared by the submission day.


### Monitor

You may monitor experiments (except from prior skills learning) with (given that SPiRL modification didn't break it)
```
tensorboard --logdir=/path/to/experiment
```

## Logging

By default the checkpointing will be in your home directory. You can change it by defining a `CHECKPOINT` environment variable. Add the following to your `.bashrc` file to change the logging directory.
```
export CHECKPOINT=/path/to/checkpoints
```

## Cite

Please cite the original work if you use this code or compare to this approach
```
@inproceedings{strudelnmp2020,
title={Learning Obstacle Representations for Neural Motion Planning},
author={R. {Strudel} and R. {Garcia} and J. {Carpentier} and J.P. {Laumond} and I. {Laptev} and C. {Schmid}},
journal={Proceedings of Conference on Robot Learning (CoRL)},
year={2020}
}
```

import jax as jx
import jax.numpy as jnp
import numpy as np

import argparse

import pickle as pkl

import gym

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

#add parent dir to import path
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dreamer import dreamer_agent

min_denom = 0.000001

action_map = ['n', 'l', 'u', 'r', 'd', 'f']

########################################################################
# Parse Arguments and Config File
########################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=str, default="dreamer.out")
parser.add_argument("--model", "-m", type=str, default="dreamer.model")
parser.add_argument("--seed", "-s", type=int, default=0)
parser.add_argument("--episodes", "-e", type=int, default=10)
parser.add_argument("--autoplay", "-a", action='store_true', default=False)
args = parser.parse_args()
key = jx.random.PRNGKey(args.seed)

with open(args.model, 'rb') as f:
    params = pkl.load(f)
    print("loaded model params from " + args.model)

with open(args.output, 'rb') as f:
    data = pkl.load(f)
    print("loaded data params from " + args.output)
    config = data['config']

env = gym.make(config['env'])
num_actions = env.action_space.n
n_channels = env.observation_space.shape[2]
cmap = sns.color_palette("cubehelix", n_channels)
cmap.insert(0, (0, 0, 0))
cmap = colors.ListedColormap(cmap)
bounds = [i for i in range(n_channels + 2)]
norm = colors.BoundaryNorm(bounds, n_channels + 1)
_, ax = plt.subplots(1, 1)
plt.show(block=False)

def display_state(state):
    numerical_state = np.amax(
        state * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2) + 0.5
    ax.imshow(numerical_state, cmap=cmap, norm=norm, interpolation='none')
    if(args.autoplay):
        plt.pause(50/1000)
    else:
        plt.waitforbuttonpress()
    plt.cla()


def play(agent, env):
    for i in range(args.episodes):
        state = env.reset()
        terminal = False
        G = 0.0
        while(not terminal):
            display_state(state)
            a = agent.act(state.astype(float))
            state, reward, terminal, _ = env.step(a)
            G += reward
            # print('value: ' + str(agent.current_value(state)))
            print('reward: '+str(reward))
        print("Return: "+str(G))




key, subkey = jx.random.split(key)

agent = dreamer_agent(subkey, env, config, params=params)


play(agent, env)

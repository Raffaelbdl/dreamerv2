import jax as jx
import jax.numpy as jnp
import numpy as np

import argparse

import pickle as pkl

import gym

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

# add parent dir to import path
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dreamer import dreamer_agent, log_binary_probability

min_denom = 0.000001

action_map = ['n', 'l', 'u', 'r', 'd', 'f']

########################################################################
# Parse Arguments and Config File
########################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=str, default="dreamer.out")
parser.add_argument("--model", "-m", type=str, default="dreamer.model")
parser.add_argument("--buffer", "-b", type=str, default="dreamer.buffer")
parser.add_argument("--init_length", "-i", type=int, default=5)
parser.add_argument("--seed", "-s", type=int, default=0)
parser.add_argument("--rollout_length", "-r", type=int, default=10)
parser.add_argument("--autoplay", "-a", action='store_true', default=False)

args = parser.parse_args()
key = jx.random.PRNGKey(args.seed)

with open(args.model, 'rb') as f:
    params = pkl.load(f)
    print("loaded model params from " + args.model)

with open(args.output, 'rb') as f:
    data = pkl.load(f)
    print("loaded data from " + args.output)
    config = data['config']

with open(args.buffer, 'rb') as f:
    buffers = pkl.load(f)

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
        plt.pause(50 / 1000)
    else:
        plt.waitforbuttonpress()
    plt.cla()


def rollout_model(agent, obervations, actions, key):
    model_params = agent.model_params()
    model_networks = agent.model_networks

    reward_params = model_params['reward']
    termination_params = model_params['termination']
    recurrent_params = model_params['recurrent']
    phi_params = model_params['phi']
    next_phi_params = model_params['next_phi']
    state_params = model_params['state']

    reward_network = model_networks['reward']
    termination_network = model_networks['termination']
    recurrent_network = model_networks['recurrent']
    phi_network = model_networks['phi']
    next_phi_network = model_networks['next_phi']
    state_network = model_networks['state']

    actor_network = agent.actor_apply
    actor_params = agent.actor_params()

    h = jnp.zeros((1, config['num_hidden_units']))
    for i in range(args.init_length):
        o = observations[:, i]
        a = actions[:, i]
        key, subkey = jx.random.split(key)
        phi, _ = phi_network(phi_params, o, h, subkey)
        one_hot_action = jnp.eye(num_actions)[a]
        h = recurrent_network(recurrent_params, phi, one_hot_action, h)

    for t in range(args.rollout_length):
        key, subkey = jx.random.split(key)
        curr_pi_logit = actor_network(actor_params, phi, h)

        key, subkey = jx.random.split(key)
        action = jx.random.categorical(subkey, curr_pi_logit)
        one_hot_action = jnp.eye(num_actions)[action]

        h = recurrent_network(recurrent_params, phi, one_hot_action, h)

        phi, _ = next_phi_network(next_phi_params, h, subkey)

        key, subkey = jx.random.split(key)
        reward_dist_params = reward_network(reward_params, phi, h)
        reward = reward_dist_params['mu']

        key, subkey = jx.random.split(key)
        gamma_dist_params = termination_network(termination_params, phi, h)
        gamma = jnp.exp(log_binary_probability(True, gamma_dist_params))

        S_hat_params = state_network(state_params, phi, h)
        key, subkey = jx.random.split(key)
        S = jx.random.bernoulli(key, jx.nn.sigmoid(S_hat_params['logit']))
        print('action: ' + str(action_map[env.action_set[action[0]]]))
        print('reward: ' + str(reward[0]))
        print('gamma: ' + str(gamma[0]))
        display_state(S[0])


key, subkey = jx.random.split(key)
agent = dreamer_agent(subkey, env, config, params=params)
agent.set_buffers(buffers)
key, subkey = jx.random.split(key)

observations, actions, _, _ = agent.buffer.sample(1, args.init_length)

print('Showing start state.')
display_state(observations[-1][0])
print('starting rollout.')
rollout_model(agent, observations, actions, subkey)

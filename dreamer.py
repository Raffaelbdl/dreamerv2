import jax as jx
import jax.numpy as jnp

import numpy as np

from jax import grad, jit, vmap

import haiku as hk

import json

import argparse

import time

from optimizers import adamw

from jax.experimental import optimizers

import networks as nets

import pickle as pkl

import copy

import gym

min_denom = 0.000001

########################################################################
# Probability Helper Functions
########################################################################
def log_gaussian_probability(x, params):
    mu = params['mu']
    sigma = params['sigma']
    return -(jnp.log(sigma) + 0.5 * jnp.log(2 * jnp.pi) + 0.5 * ((x - mu) / sigma)**2)


def gaussian_cross_entropy(params_1, params_2):
    mu_1 = params_1['mu']
    sigma_1 = params_1['sigma']
    mu_2 = params_2['mu']
    sigma_2 = params_2['sigma']
    return 0.5 * jnp.log(2 * jnp.pi) + jnp.log(sigma_2) + (sigma_1**2 + (mu_1 - mu_2)**2) / (2 * sigma_2**2)


def gaussian_entropy(params):
    sigma = params['sigma']
    return 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(sigma)


def gaussian_KL(params_1, params_2):
    return gaussian_cross_entropy(params_1, params_2) - gaussian_entropy(params_1)


def log_binary_probability(x, params):
    logit = params['logit']
    return jnp.where(x, jx.nn.log_sigmoid(logit), jx.nn.log_sigmoid(-logit))


def binary_entropy(params):
    logit = params['logit']
    return jx.nn.sigmoid(logit) * jx.nn.log_sigmoid(logit) + jx.nn.sigmoid(-logit) * jx.nn.log_sigmoid(-logit)


def categorical_cross_entropy(params_1, params_2):
    probs_1 = params_1['probs']
    log_probs_2 = params_2['log_probs']
    return -jnp.sum(probs_1 * log_probs_2, axis=(-1))


def categorical_entropy(params):
    probs = params['probs']
    log_probs = params['log_probs']
    return -jnp.sum(probs * log_probs, axis=(-1))


def categorical_KL(params_1, params_2):
    return categorical_cross_entropy(params_1, params_2) - categorical_entropy(params_1)

########################################################################
# Define Replay Buffer
########################################################################

# draws a sequence of length sequence length from each data buffer begining at i, except shifted to extend backward as much as possible to maximize sequence length within episode
def draw_sequence(i, sequence_length, max_index, buffer_end_location, observations, actions, rewards, terminals):
    # draw terminal sequence, wrapping at end of replay buffer
    terms = jnp.take(terminals,jnp.arange(sequence_length)+i, axis=0, mode='wrap')
    # find first terminal in sequence (if any)
    first_terminal_index= jnp.nonzero(terms,size=1, fill_value=-1)[0]

    # shift i such that the sequence ends at the first terminal index (if any are present)
    i = jnp.where(first_terminal_index!=-1,(i-(sequence_length-1-first_terminal_index))%max_index, i)

    # draw terminals in new sequence
    terms = jnp.take(terminals,jnp.arange(sequence_length)+i, axis=0, mode='wrap')
    # find last terminal in sequence besides the one found in the last step (if any others are present)
    last_terminal_index = jnp.nonzero(jnp.flip(terms[:-1]),size=1, fill_value=-1)[0]

    # shift i to just after the second last terminal sequence (only if we found a terminal in the first step and another in the second)
    i = jnp.where(jnp.logical_and(first_terminal_index!=-1,last_terminal_index!=-1),(i+sequence_length-last_terminal_index-1)%max_index,i)

    end_index = i+sequence_length

    # check if the end of the buffer is included in the interval (note this may result in the occasional shorter sequence but this shouldn't matter much)
    buffer_end_in_interval = jnp.where(end_index <= max_index, jnp.logical_and(i <= buffer_end_location, buffer_end_location<end_index), jnp.logical_or(buffer_end_location<end_index-max_index,buffer_end_location>=i))

    # shift sequences so they do not include the end of the buffer
    i = jnp.where(buffer_end_in_interval,(buffer_end_location-sequence_length)%max_index,i)

    # sample other indices
    obs = jnp.take(observations,jnp.arange(sequence_length)+i, axis=0, mode='wrap')
    acts = jnp.take(actions,jnp.arange(sequence_length)+i, axis=0, mode='wrap')
    rs = jnp.take(rewards,jnp.arange(sequence_length)+i, axis=0, mode='wrap')
    terms = jnp.take(terminals,jnp.arange(sequence_length)+i, axis=0, mode='wrap')

    return obs, acts, rs, terms

# transform single sequence draw into batched versoin, only apply vmap to input indices
draw_sequences = jit(vmap(draw_sequence, in_axes=(0,None,None,None,None,None,None,None)), static_argnums=(1))

# numpy implementation of draw_sequences for case where replay buffer must be stored in RAM, this could be better optimized
def draw_sequences_np(indices, sequence_length, max_index, buffer_end_location, observations, actions, rewards, terminals):
    all_obs = []
    all_acts = []
    all_rs = []
    all_terms = []
    # just sample indices sequentially here
    for i in indices:
        # draw terminal sequence, wrapping at end of replay buffer
        terms = np.take(terminals,np.arange(sequence_length)+i, axis=0, mode='wrap')
        # find first terminal in sequence (if any)
        terminal_indices = np.nonzero(terms)[0]
        first_terminal_index= -1 if len(terminal_indices)==0 else terminal_indices[0]

        # shift i such that the sequence ends at the first terminal index (if any are present)
        i = np.where(first_terminal_index!=-1,(i-(sequence_length-1-first_terminal_index))%max_index, i)

        # draw terminals in new sequence
        terms = np.take(terminals,np.arange(sequence_length)+i, axis=0, mode='wrap')
        # find last terminal in sequence besides the one found in the last step (if any others are present)
        other_terminal_indices = np.nonzero(np.flip(terms[:-1]))[0]
        last_terminal_index = -1 if len(other_terminal_indices)==0 else other_terminal_indices[0]

        # shift i to just after the second last terminal sequence (only if we found a terminal in the first step and another in the second)
        i = np.where(np.logical_and(first_terminal_index!=-1,last_terminal_index!=-1),(i+sequence_length-last_terminal_index-1)%max_index,i)

        end_index = i+sequence_length

        # check if the end of the buffer is included in the interval (note this may result in the occasional shorter sequence but this shouldn't matter much)
        buffer_end_in_interval = np.where(end_index <= max_index, np.logical_and(i <= buffer_end_location, buffer_end_location<end_index), np.logical_or(buffer_end_location<end_index-max_index,buffer_end_location>=i))

        # shift sequences so they do not include the end of the buffer
        i = np.where(buffer_end_in_interval,(buffer_end_location-sequence_length)%max_index,i)

        # sample other indices
        all_obs += [np.take(observations,np.arange(sequence_length)+i, axis=0, mode='wrap')]
        all_acts += [np.take(actions,np.arange(sequence_length)+i, axis=0, mode='wrap')]
        all_rs += [np.take(rewards,np.arange(sequence_length)+i, axis=0, mode='wrap')]
        all_terms += [np.take(terminals,np.arange(sequence_length)+i, axis=0, mode='wrap')]
    return np.stack(all_obs), np.stack(all_acts), np.stack(all_rs), np.stack(all_terms)



class replay_buffer:
    def __init__(self, buffer_size, obs_shape, key, use_cpu=False):
        self.key = key
        self.buffer_size = buffer_size
        self.location = 0
        self.full = False
        self.use_cpu = use_cpu

        # if use_cpu is set replay buffer uses ordinary numpy arrays, otherwise jax.numpy arrays
        if(self.use_cpu):
            self.observations = np.zeros([buffer_size] + list(obs_shape), dtype=float)
            self.actions = np.zeros(buffer_size, dtype=int)
            self.rewards = np.zeros(buffer_size, dtype=float)
            self.terminals = np.zeros(buffer_size, dtype=bool)
        else:
            self.observations = jnp.zeros([buffer_size] + list(obs_shape), dtype=float)
            self.actions = jnp.zeros(buffer_size, dtype=int)
            self.rewards = jnp.zeros(buffer_size, dtype=float)
            self.terminals = jnp.zeros(buffer_size, dtype=bool)

    def add(self, obs, action, reward, terminal):
        if(self.use_cpu):
            self.observations[self.location] = np.asarray(obs, dtype=float)
            self.actions[self.location] = np.asarray(action, dtype=int)
            self.rewards[self.location] = np.asarray(reward, dtype=float)
            self.terminals[self.location] = np.asarray(terminal, dtype=bool)
        else:
            self.observations = self.observations.at[self.location].set(jnp.asarray(obs, dtype=float))
            self.actions = self.actions.at[self.location].set(jnp.asarray(action, dtype=int))
            self.rewards = self.rewards.at[self.location].set(jnp.asarray(reward, dtype=float))
            self.terminals = self.terminals.at[self.location].set(jnp.asarray(terminal, dtype=bool))
        if self.location == self.buffer_size - 1:
            self.full = True
        # increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size, sequence_length):
        max_index = self.buffer_size if self.full else self.location

        self.key, subkey = jx.random.split(self.key)
        start_indices = jx.random.choice(subkey, max_index, shape=(batch_size,))
        if(self.use_cpu):
            return draw_sequences_np(start_indices, sequence_length, max_index, self.location, self.observations, self.actions, self.rewards, self.terminals)
        else:
            return draw_sequences(start_indices, sequence_length, max_index, self.location, self.observations, self.actions, self.rewards, self.terminals)

    # useful for pickling to a file
    def get_buffers(self):
        return {'observations': self.observations, 'actions': self.actions, 'rewards': self.rewards, 'terminals': self.terminals}

    def set_buffers(self, buffers):
        # assume that buffers being set are full since we have no way to tell apriori
        self.observations = buffers['observations']
        self.actions = buffers['actions']
        self.rewards = buffers['rewards']
        self.terminals = buffers['terminals']
        self.location = 0
        self.full = True
        self.buffer_size = self.observations.shape[0]

########################################################################
# Define Losses and Other Functions
########################################################################
def batchwise_flatten(x):
    return jnp.reshape(x, [x.shape[0],-1])

#This returns the model loss along with intermediate agent states which are then reused to initialize the actor-critic training
def get_model_loss_and_states_function(networks, binary_state, image_state, num_actions):
    def model_loss_and_states(params, key, observations, actions, rewards, terminals):
        recurrent_params = params['recurrent']
        phi_params = params['phi']
        next_phi_params = params['next_phi']
        reward_params = params['reward']
        termination_params = params['termination']


        recurrent_network = networks['recurrent']
        phi_network = networks['phi']
        next_phi_network = networks['next_phi']
        reward_network = networks['reward']
        termination_network = networks['termination']

        if(state_prediction_weight>0.0):
            state_network = networks['state']
            state_params = params['state']

        h_list = []
        phi_list = []

        # initialize hidden state for recurrent network
        h = jnp.zeros((batch_size, num_hidden_units))
        loss = 0.0
        phi_hat_dist = None
        r_dist = None
        gamma_dist = None
        # record whether each batch element has terminated
        terminated = jnp.zeros(batch_size)
        nonterminal_steps = 0
        # generate sequential predictions
        for i in range(sequence_length):
            key, subkey = jx.random.split(key)
            phi, phi_dist = phi_network(phi_params, observations[:,i], h, subkey)

            h_list+=[h]
            phi_list+=[phi]

            one_hot_actions = jnp.eye(num_actions)[actions[:,i]]

            if(state_prediction_weight>0.0):
                S_hat_params = state_network(state_params, phi, h)
                if(binary_state):
                    S_log_probs = log_binary_probability(observations[:,i], S_hat_params)
                else:
                    S_log_probs = log_gaussian_probability(observations[:,i], S_hat_params)
                if(image_state):
                    log_P_S = jnp.sum(S_log_probs, axis=(1,2,3))
                else:
                    log_P_S = jnp.sum(S_log_probs, axis=(1))
                state_prediction_loss = -log_P_S
            else:
                state_prediction_loss = 0.0

            # no need to reconstruct state on terminal steps, just need to get reward and terminal right
            state_prediction_loss = jnp.where(terminals[:,i], 0.0, state_prediction_loss)

            r_dist = reward_network(reward_params, phi, h)
            reward_loss = -log_gaussian_probability(rewards[:,i], r_dist)

            gamma_dist = termination_network(termination_params, phi, h)
            termination_loss = -log_binary_probability(jnp.logical_not(terminals[:,i]), gamma_dist)

            key, subkey = jx.random.split(key)
            phi_hat, phi_hat_dist = next_phi_network(next_phi_params, h, subkey)
            # KL loss applied to make current phi closer to prediction
            KL_posterior_loss = jnp.sum(latent_KL(phi_dist,jx.lax.stop_gradient(phi_hat_dist)),axis=1)

            posterior_entropy_loss = -jnp.sum(latent_entropy(phi_dist),axis=1)

            # KL loss applied to make prediction closer to current phi
            KL_prior_loss = jnp.sum(latent_KL(jx.lax.stop_gradient(phi_dist),phi_hat_dist),axis=1)

            step_loss = (KL_posterior_weight*KL_posterior_loss+
                    KL_prior_weight*KL_prior_loss+
                    posterior_entropy_weight*posterior_entropy_loss+
                    reward_weight*reward_loss+
                    termination_weight*termination_loss+
                    state_prediction_weight*state_prediction_loss)

            # no need to predict anything that occurs after termination
            step_loss = jnp.where(terminated, 0.0, step_loss)

            # Note loss is summed over batches and sequence steps
            loss += jnp.sum(step_loss)

            h = recurrent_network(recurrent_params,phi,one_hot_actions,h)

            nonterminal_steps+=jnp.sum(jnp.logical_not(terminated))

            # record whether each batch element has terminated
            terminated = jnp.logical_or(terminated, terminals[:,i])

        # concatenate phi and h from each step along batch dimension
        h_array = jnp.concatenate(h_list,axis=0)
        phi_array = jnp.concatenate(phi_list,axis=0)

        # loss here is divided by total nonterminal steps (summed over batches)
        return loss/nonterminal_steps, phi_array, h_array
    return model_loss_and_states

# takes a function which returns loss, phis, hs and transforms it to a function which returns grad(loss), phis, hs
def model_loss_and_states_to_model_grad_and_states(func):
    def model_loss(*args):
        return func(*args)[0]

    def model_grad_and_states(*args):
        return grad(model_loss)(*args), *func(*args)[1:]

    return model_grad_and_states

def get_model_eval_function(networks, binary_state, image_state, num_actions):
    def model_eval(params, key, observations, actions, rewards, terminals):
        recurrent_params = params['recurrent']
        phi_params = params['phi']
        next_phi_params = params['next_phi']
        reward_params = params['reward']
        termination_params = params['termination']


        recurrent_network = networks['recurrent']
        phi_network = networks['phi']
        next_phi_network = networks['next_phi']
        reward_network = networks['reward']
        termination_network = networks['termination']

        if(state_prediction_weight>0.0):
            state_network = networks['state']
            state_params = params['state']

        r_0_count = 0
        r_1_count = 0

        gamma_0_count = 0
        gamma_1_count = 0

        gamma_hat_0_tot = 0.0
        gamma_hat_1_tot = 0.0
        r_hat_0_tot = 0.0
        r_hat_1_tot = 0.0

        phi_mean_cross_entropy = 0.0
        phi_mean_entropy = 0.0

        S_mean_logprob_tot = 0.0
        S_nonzero_tot = 0

        nonterminal_steps = 0

        # initialize hidden state for recurrent network
        h = jnp.zeros((batch_size, num_hidden_units))
        phi_hat_dist = None
        r_dist = None
        gamma_dist = None
        # record whether each batch element has terminated
        terminated = jnp.zeros(batch_size, dtype=bool)
        # generate sequential predictions
        for i in range(sequence_length):
            key, subkey = jx.random.split(key)
            phi, phi_dist = phi_network(phi_params, observations[:,i], h, subkey)

            key, subkey = jx.random.split(key)
            phi_hat, phi_hat_dist = next_phi_network(next_phi_params, h, subkey)

            one_hot_actions = jnp.eye(num_actions)[actions[:,i]]

            if(state_prediction_weight>0.0):
                S_hat_params = state_network(state_params, phi, h)
                if(binary_state):
                    S_log_probs = log_binary_probability(observations[:,i], S_hat_params)
                else:
                    S_log_probs = log_gaussian_probability(observations[:,i], S_hat_params)

                if(image_state):
                    log_P_S = jnp.mean(S_log_probs, axis=(1,2,3))
                    S_nonzero = jnp.mean(observations[:,i], axis=(1,2,3))
                else:
                    log_P_S = jnp.mean(S_log_probs, axis=(1))
                    S_nonzero = jnp.mean(observations[:,i], axis=(1))
                S_mean_logprob_tot += jnp.sum(jnp.where(terminated, 0.0,log_P_S))
                S_nonzero_tot += jnp.sum(jnp.where(terminated, 0.0,S_nonzero))

            r_dist = reward_network(reward_params, phi, h)
            r_hat = r_dist['mu']

            gamma_dist = termination_network(termination_params, phi, h)
            gamma_hat = jnp.exp(log_binary_probability(1.0,gamma_dist))

            phi_mean_cross_entropy += jnp.sum(jnp.where(terminated, 0.0,jnp.mean(latent_cross_entropy(phi_dist,phi_hat_dist), axis=1)))
            phi_mean_entropy += jnp.sum(jnp.where(terminated, 0.0, jnp.mean(latent_entropy(phi_dist), axis=1)))

            r_1_count += jnp.sum(jnp.where(terminated, 0.0, rewards[:,i]==1.0))
            r_0_count += jnp.sum(jnp.where(terminated, 0.0, rewards[:,i]==0.0))

            r_hat_1 = jnp.where(rewards[:,i]==1.0, r_hat, 0.0)
            r_hat_0 = jnp.where(rewards[:,i]==0.0, r_hat, 0.0)
            r_hat_1_tot += jnp.sum(jnp.where(terminated, 0.0, r_hat_1))
            r_hat_0_tot += jnp.sum(jnp.where(terminated, 0.0, r_hat_0))

            gamma_1_count += jnp.sum(jnp.where(terminated, 0.0, jnp.logical_not(terminals[:,i])))
            gamma_0_count += jnp.sum(jnp.where(terminated, 0.0, terminals[:,i]))

            gamma_hat_1 = jnp.where(jnp.logical_not(terminals[:,i]), gamma_hat, 0.0)
            gamma_hat_0 = jnp.where(terminals[:,i], gamma_hat,0.0)
            gamma_hat_1_tot += jnp.sum(jnp.where(terminated, 0.0, gamma_hat_1))
            gamma_hat_0_tot += jnp.sum(jnp.where(terminated, 0.0, gamma_hat_0))

            h = recurrent_network(recurrent_params,phi,one_hot_actions,h)

            nonterminal_steps+=jnp.sum(jnp.logical_not(terminated))

            # record whether each batch element has terminated
            terminated = jnp.logical_or(terminated, terminals[:,i])

        # reward prediction related metrics here only really make sense in MinAtar where rewards are usually 0 or 1
        metrics={'gamma_0_pred' : gamma_hat_0_tot/gamma_0_count,
                 'gamma_1_pred' : gamma_hat_1_tot/gamma_1_count,
                 'r_0_pred' : r_hat_0_tot/r_0_count,
                 'r_1_pred' : r_hat_1_tot/r_1_count,
                 'gamma_0_count': gamma_0_count,
                 'gamma_1_count': gamma_1_count,
                 'r_0_count': r_0_count,
                 'r_1_count': r_1_count,
                 'phi_cross_entropy' : phi_mean_cross_entropy/nonterminal_steps,
                 'phi_entropy' : phi_mean_entropy/nonterminal_steps,
                 'S_logprob' : S_mean_logprob_tot/nonterminal_steps,
                 'S_nonzero_tot' : S_nonzero_tot/nonterminal_steps
                 }
        return metrics
    return model_eval

def get_AC_loss_function(actor_network, critic_network, model_networks, num_actions):
    def AC_loss(actor_params, fast_critic_params, slow_critic_params, model_params, key, phis, hs):
        reward_params = model_params['reward']
        recurrent_params = model_params['recurrent']
        termination_params = model_params['termination']
        next_phi_params = model_params['next_phi']

        reward_network = model_networks['reward']
        recurrent_network = model_networks['recurrent']
        termination_network = model_networks['termination']
        next_phi_network = model_networks['next_phi']

        loss = 0.0

        curr_pi_logit = actor_network(actor_params, phis, hs)
        fast_curr_V = critic_network(fast_critic_params, phis, hs)
        slow_curr_V = critic_network(slow_critic_params, phis, hs)
        pi_logits =[]
        f_Vs = []
        s_Vs = []
        gammas = []
        rewards = []
        actions = []
        for t in range(rollout_length):
            f_Vs+=[fast_curr_V]
            s_Vs+=[slow_curr_V]
            pi_logits+=[curr_pi_logit]

            key, subkey = jx.random.split(key)
            action = jx.random.categorical(subkey, curr_pi_logit)
            one_hot_action = jnp.eye(num_actions)[action]
            actions+=[one_hot_action]

            hs = recurrent_network(recurrent_params,phis,one_hot_action,hs)

            key, subkey = jx.random.split(key)
            phis, _ = jx.lax.stop_gradient(next_phi_network(next_phi_params, hs, subkey))

            # just use mean reward when sampling
            reward_dist = reward_network(reward_params, phis, hs)
            reward = reward_dist['mu']
            rewards += [reward]

            gamma_dist = termination_network(termination_params, phis, hs)
            gamma = jnp.exp(log_binary_probability(True, gamma_dist))*discount
            gammas += [gamma]

            curr_pi_logit = actor_network(actor_params, phis, hs)
            fast_curr_V = critic_network(fast_critic_params, phis, hs)
            slow_curr_V = critic_network(slow_critic_params, phis, hs)

        G = reward+gamma*jx.lax.stop_gradient(slow_curr_V)
        for pi_logit, f_V, s_V, action, gamma, reward in zip(reversed(pi_logits), reversed(f_Vs), reversed(s_Vs), reversed(actions), reversed(gammas), reversed(rewards)):
            critic_loss = jnp.mean(0.5*(G-f_V)**2)
            entropy = -jnp.sum(jx.nn.log_softmax(pi_logit)*jx.nn.softmax(pi_logit), axis=1)

            actor_loss = jnp.mean(-0.5*jx.lax.stop_gradient(G-s_V)*jnp.sum(jx.nn.log_softmax(pi_logit)*action,axis=1)-beta*entropy)

            G = reward+gamma*((1-lmbda)*jx.lax.stop_gradient(s_V)+lmbda*G)

            loss += jnp.mean(critic_loss+actor_loss)

        return loss/rollout_length
    return AC_loss

def get_act_function(actor_network, recurrent_network, num_actions):
    def act(actor_params, recurrent_params, phi, h, key, action, sample_action):
        pi_logit = actor_network(actor_params, phi, h)

        if(sample_action):
            key, subkey = jx.random.split(key)
            a = jx.random.categorical(subkey, pi_logit)
        else:
            a = jnp.expand_dims(action,axis=0)

        h = recurrent_network(recurrent_params,phi,jnp.eye(num_actions)[a],h)
        return a, h
    return act

def get_observe_function(phi_network):
    def observe(phi_params, observation, phi, h, key):
        key, subkey = jx.random.split(key)
        phi, _ = phi_network(phi_params, observation, h, subkey)
        return phi
    return observe


########################################################################
# Define Agent
########################################################################
class dreamer_agent():
    def __init__(self, key, env, config, params=None):
        self.beta = config['beta']
        self.t = 0

        self.key = key

        self.config = config

        self.valid_actions = list(range(env.action_space.n))

        num_actions = len(self.valid_actions)

        state_shape = env.observation_space.sample().shape
        if(env.observation_space.sample().dtype==bool):
            self.binary_state = True
        else:
            self.binary_state = False

        # Model related initialization
        #============================
        model_opt_init, self.model_opt_update, self.get_model_params = adamw(config['model_alpha'], eps=config['eps_adam'], wd=config['wd_adam'])
        model_params = {}
        self.model_networks = {}

        # dummy variables for network initialization
        self.key, dummy_key = jx.random.split(self.key)
        dummy_state = jnp.zeros([1]+list(state_shape))
        dummy_phi = jnp.zeros((1,config['num_features']*(config['feature_width'] if config['latent_type']=='categorical' else 1)))
        dummy_a = jnp.zeros((1,num_actions))
        dummy_h = jnp.zeros((1,config['num_hidden_units']))

        self.key, subkey = jx.random.split(self.key)
        self.buffer = replay_buffer(config['buffer_size'], state_shape, subkey, config['cpu_replay'])

        # initialize recurrent network
        recurrent_net = hk.without_apply_rng(hk.transform(lambda phi, a, h: nets.recurrent_network(config)(phi,a,h)))
        self.key, subkey = jx.random.split(self.key)
        recurrent_params = recurrent_net.init(subkey,dummy_phi, dummy_a, dummy_h)
        recurrent_apply = recurrent_net.apply
        model_params['recurrent']=recurrent_params
        self.model_networks['recurrent']=recurrent_apply

        # initialize phi network
        if(len(state_shape)>1 and not config['no_conv']):
            image_state = True
            phi_net = hk.without_apply_rng(hk.transform(lambda s, h, k: nets.phi_conv_network(config)(s, h, k)))
        else:
            image_state = False
            phi_net = hk.without_apply_rng(hk.transform(lambda s, h, k: nets.phi_flat_network(config)(s, h, k)))
        phi_apply = phi_net.apply
        self.key, subkey = jx.random.split(self.key)
        phi_params = phi_net.init(subkey,dummy_state, dummy_h, dummy_key)
        model_params['phi']=phi_params
        self.model_networks['phi']=phi_apply

        # initialize reward network
        reward_net = hk.without_apply_rng(hk.transform(lambda phi, h: nets.reward_network(config)(phi, h)))
        self.key, subkey = jx.random.split(self.key)
        reward_params = reward_net.init(subkey, dummy_phi, dummy_h)
        reward_apply = reward_net.apply
        model_params['reward']=reward_params
        self.model_networks['reward']=reward_apply

        # initialize termination network
        termination_net = hk.without_apply_rng(hk.transform(lambda phi, h: nets.termination_network(config)(phi, h)))
        self.key, subkey = jx.random.split(self.key)
        termination_params = termination_net.init(subkey, dummy_phi, dummy_h)
        termination_apply = termination_net.apply
        model_params['termination']=termination_params
        self.model_networks['termination']=termination_apply

        # initialize phi prediction network
        next_phi_net = hk.without_apply_rng(hk.transform(lambda h, k: nets.next_phi_network(config)(h, k)))
        next_phi_apply = next_phi_net.apply
        self.key, subkey = jx.random.split(self.key)
        next_phi_params = next_phi_net.init(subkey, dummy_h, dummy_key)
        model_params['next_phi']=next_phi_params
        self.model_networks['next_phi']=next_phi_apply

        # initialize state reconstruction network
        if(config['state_prediction_weight']>0.0):
            if(image_state):
                state_net = hk.without_apply_rng(hk.transform(lambda phi, h: nets.state_conv_network(config, self.binary_state, state_shape)(phi, h)))
            else:
                state_width = 1
                for j in state_shape:
                    state_width*=j
                state_net = hk.without_apply_rng(hk.transform(lambda phi, h: nets.state_flat_network(config, self.binary_state, state_width)(phi, h)))
            state_apply = state_net.apply
            self.key, subkey = jx.random.split(self.key)
            state_params = state_net.init(subkey, dummy_phi, dummy_h)
            model_params['state']=state_params
            self.model_networks['state']=state_apply

        if(params is not None):
            model_params = params["model"]

        self.model_opt_state = model_opt_init(model_params)
        self.model_opt_update = jit(self.model_opt_update)

        model_loss_and_states = get_model_loss_and_states_function(self.model_networks, self.binary_state, image_state, num_actions)

        # this returns model_grads, model_states
        self.model_grad_and_states = jit(model_loss_and_states_to_model_grad_and_states(model_loss_and_states))

        self.model_eval = jit(get_model_eval_function(self.model_networks, self.binary_state, image_state, num_actions))

        # AC related initialization
        #=========================
        actor_opt_init, self.actor_opt_update, self.get_actor_params = adamw(config['actor_alpha'], eps=config['eps_adam'], wd=config['wd_adam'])
        critic_opt_init, self.critic_opt_update, self.get_critic_params = adamw(config['critic_alpha'], eps=config['eps_adam'], wd=config['wd_adam'])

        # initialize actor network
        actor_net = hk.without_apply_rng(hk.transform(lambda phi, h: nets.actor_network(config, num_actions)(phi, h)))
        self.actor_apply = actor_net.apply
        self.key, subkey = jx.random.split(self.key)
        actor_params = actor_net.init(subkey, dummy_phi, dummy_h)

        # initialize fast critic network
        critic_net = hk.without_apply_rng(hk.transform(lambda phi, h: nets.critic_network(config)(phi, h)))
        self.critic_apply = critic_net.apply
        self.key, subkey = jx.random.split(self.key)
        fast_critic_params = critic_net.init(subkey ,dummy_phi, dummy_h)

        if(params is not None):
            fast_critic_params = params["critic"]
            actor_params = params["actor"]

        self.slow_critic_params = copy.deepcopy(fast_critic_params)

        self.actor_opt_state = actor_opt_init(actor_params)
        self.actor_opt_update = jit(self.actor_opt_update)
        self.critic_opt_state = critic_opt_init(fast_critic_params)
        self.critic_opt_update = jit(self.critic_opt_update)
        self.AC_loss_grad = jit(grad(get_AC_loss_function(self.actor_apply,self.critic_apply,self.model_networks, num_actions),argnums=(0,1)))

        # maintain state information for acting in the real world
        self.h = jnp.zeros(dummy_h.shape)
        self.phi = jnp.zeros(dummy_phi.shape)

        # phi_params, phi_network, observation, phi, h, key
        self._observe = jit(get_observe_function(self.model_networks['phi']))
        # actor_params, recurrent_params, phi, h, key, action, sample_action
        self._act = jit(get_act_function(self.actor_apply,self.model_networks['recurrent'],num_actions),static_argnames=('sample_action'))

    def act(self, observation, random=False):
        self.key, subkey = jx.random.split(self.key)
        self.phi = self._observe(self.model_params()['phi'], jnp.expand_dims(jnp.asarray(observation,dtype=float), axis=0), self.phi, self.h, subkey)

        if(random):
            self.key, subkey = jx.random.split(self.key)
            action = int(jx.random.choice(subkey, jnp.array(self.valid_actions)))
        else:
            action = 0

        self.key, subkey = jx.random.split(self.key)
        action, self.h = self._act(self.actor_params(), self.model_params()['recurrent'], self.phi, self.h, subkey, action, not random)

        return self.valid_actions[int(action)]

    def reset(self):
        self.h = jnp.zeros(self.h.shape)
        self.phi = jnp.zeros(self.phi.shape)

    def model_params(self):
        return self.get_model_params(self.model_opt_state)

    def actor_params(self):
        return self.get_actor_params(self.actor_opt_state)

    def critic_params(self):
        return self.get_critic_params(self.critic_opt_state)

    def add_to_replay(self, *args):
        self.buffer.add(*args)

    def get_buffers(self):
        return self.buffer.get_buffers()

    def set_buffers(self, buffers):
        self.buffer.set_buffers(buffers)

    def update(self):
        observations, actions, rewards, terminals = self.buffer.sample(self.config['batch_size'], self.config['sequence_length'])
        self.key, subkey = jx.random.split(self.key)
        grads, phis, hs = self.model_grad_and_states(self.model_params(), subkey, observations, actions, rewards, terminals)
        grads = optimizers.clip_grads(grads, self.config['grad_clip'])
        self.model_opt_state = self.model_opt_update(self.t, grads, self.model_opt_state)

        self.key, subkey = jx.random.split(self.key)
        grads = self.AC_loss_grad(self.actor_params(), self.critic_params(), self.slow_critic_params, self.model_params(), subkey, phis, hs)
        grads = optimizers.clip_grads(grads, self.config['grad_clip'])
        self.actor_opt_state = self.actor_opt_update(self.t, grads[0], self.actor_opt_state)
        self.critic_opt_state = self.critic_opt_update(self.t, grads[1], self.critic_opt_state)

        self.t += 1

    def current_value(self, observation):
        self.key, subkey = jx.random.split(self.key)
        phi = self._observe(self.model_params()['phi'], jnp.expand_dims(jnp.asarray(observation,dtype=float), axis=0), self.phi, self.h, subkey)
        return self.critic_apply(self.critic_params(), phi, self.h)

    def sync_slow_critic(self):
        self.slow_critic_params = copy.deepcopy(self.critic_params())

    def eval(self):
        observations, actions, rewards, terminals = self.buffer.sample(self.config['batch_size'], self.config['sequence_length'])
        self.key, subkey = jx.random.split(self.key)
        return self.model_eval(self.model_params(), subkey, observations, actions, rewards, terminals)

if __name__ == "__main__":
    ########################################################################
    # Parse Arguments and Config File
    ########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default="dreamer.out")
    parser.add_argument("--model", "-m", type=str, default="dreamer.model")
    parser.add_argument("--buffer", "-b", type=str, default="dreamer.buffer")
    parser.add_argument("--load", "-l", type=str, default=None)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--config", "-c", type=str)
    args = parser.parse_args()
    key = jx.random.PRNGKey(args.seed)

    if(args.load is not None):
        with open(args.load, 'rb') as f:
            params = pkl.load(f)
            print("loaded model params from "+args.load)
    else:
        params = None

    with open(args.config, 'r') as f:
        config = json.load(f)

    def get_config(k, d, default=None):
        if k not in d:
            d[k] = default
        return d[k]

    eval_frequency = get_config("eval_frequency",config,None)
    save_frequency = get_config("save_frequency",config,None)
    cpu_replay = get_config("cpu_replay", config, False)

    model_alpha = get_config("model_alpha",config,None)
    actor_alpha = get_config("actor_alpha",config,None)
    critic_alpha = get_config("critic_alpha",config,None)
    beta = get_config("beta",config,None)
    grad_clip = get_config("grad_clip",config,None)
    eps_adam = get_config("eps_adam",config,None)
    wd_adam = get_config("wd_adam",config,None)
    slow_critic_interval = get_config("slow_critic_interval",config,None)

    num_frames = get_config("num_frames",config,None)
    buffer_size = get_config("buffer_size",config,None)
    batch_size = get_config("batch_size",config,None)
    sequence_length = get_config("sequence_length",config,None)
    lmbda = get_config("lmbda",config,None)
    discount = get_config("discount",config,None)

    training_start_time = get_config("training_start_time",config,None)
    rollout_length = get_config("rollout_length",config,None)
    train_frequency = get_config("train_frequency",config,None)

    KL_prior_weight = get_config("KL_prior_weight",config,None)
    KL_posterior_weight = get_config("KL_posterior_weight",config,None)
    reward_weight = get_config("reward_weight",config,None)
    termination_weight = get_config("termination_weight",config,None)
    state_prediction_weight = get_config("state_prediction_weight",config,None)
    posterior_entropy_weight = get_config("posterior_entropy_weight", config, 0.0)

    num_features = get_config("num_features",config,None)
    feature_width = get_config("feature_width",config,None)
    num_hidden_units = get_config("num_hidden_units",config,None)

    learn_reward_variance = get_config("learn_reward_variance",config,False)
    no_conv = get_config("no_conv",config,False)
    latent_type = get_config("latent_type", config, "categorical")

    if(latent_type=='gaussian' or latent_type=='tanh_gaussian'):
        latent_KL = gaussian_KL
        latent_entropy = gaussian_entropy
        latent_cross_entropy = gaussian_cross_entropy
    elif(latent_type=='categorical'):
        latent_KL = categorical_KL
        latent_entropy = categorical_entropy
        latent_cross_entropy = categorical_cross_entropy
    else:
        raise ValueError('Unrecognized latent type.')


    ########################################################################
    # Initialize Agent and Environments
    ########################################################################
    env = gym.make(config['env'])
    key, subkey = jx.random.split(key)
    env.seed(int(subkey[0]))

    key, subkey = jx.random.split(key)
    agent = dreamer_agent(subkey, env, config, params)

    reward = 0.0
    terminal = False
    last_state = None
    returns = []
    curr_return = 0.0
    avg_return = 0.0
    termination_times = []
    evaluation_metrics = {}
    t = 0
    t_start = time.time()
    episode = 0

    ########################################################################
    # Agent Environment Interaction Loop
    ########################################################################
    while t < num_frames:
        agent.reset()
        state = env.reset()
        reward = 0.0
        terminal = False
        while(not terminal) and t < num_frames:
            # act randomly until training begins
            action = agent.act(state.astype(float), random=(t <= training_start_time))

            agent.add_to_replay(state, action, reward, terminal)

            state, reward, terminal, _ = env.step(action)

            curr_return += reward

            if(terminal):
                episode += 1
                termination_times += [t]
                avg_return = 0.99 * avg_return + 0.01 * curr_return
                returns += [curr_return]
                curr_return = 0.0

                # add terminal experience to replay immediately, action should be irrelevant since we don't take action in terminal states
                agent.add_to_replay(state, action, reward, terminal)

            if((t >= training_start_time) and (t%train_frequency==0)):
                agent.update()

            if((t >= training_start_time) and (t%slow_critic_interval==0)):
                agent.sync_slow_critic()
            t += 1


            ########################################################################
            # Logging
            ########################################################################
            if(t%eval_frequency == 0):
                metrics = agent.eval()
                print("Avg return: "+str(0.0 if episode==0 else jnp.around(avg_return/(1-0.99**episode), 2)))
                print("Frame: "+str(t))
                print("Time per frame: "+str((time.time()-t_start)/t))
                for i, k in enumerate(metrics):
                    print((k+": "+str(metrics[k])).ljust(30),end=('\n' if i>1 and (i+1)%4==0 else '| '))
                    if k in evaluation_metrics:
                        evaluation_metrics[k]+=[metrics[k]]
                    else:
                        evaluation_metrics[k]=[metrics[k]]
                print()

            # dump data every save-frequency frames
            if(t%save_frequency==0):
                # data is always dumbed to the same file and overwritten
                with open(args.output, 'wb') as f:
                    pkl.dump({
                        'config': config,
                        'returns':returns,
                        'termination_times':termination_times,
                        'evaluation_metrics':evaluation_metrics
                    }, f)

                # model is saved seperately at each save time
                with open(args.model+"_"+str(t),'wb') as f:
                    pkl.dump({
                        'model':agent.model_params(),
                        'actor': agent.actor_params(),
                        'critic': agent.critic_params()
                    }, f)


    ########################################################################
    # Save Data and Weights
    ########################################################################
    with open(args.output, 'wb') as f:
        pkl.dump({
            'config': config,
            'returns':returns,
            'termination_times':termination_times,
            'evaluation_metrics':evaluation_metrics
        }, f)

    with open(args.model,'wb') as f:
        pkl.dump({
            'model':agent.model_params(),
            'actor': agent.actor_params(),
            'critic': agent.critic_params()
        }, f)

    with open(args.buffer, 'wb') as f:
        pkl.dump(agent.get_buffers(), f)

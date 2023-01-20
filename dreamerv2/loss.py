from typing import Callable, Dict

import haiku as hk
import jax
from jax import grad, vmap, jit, value_and_grad as vgrad
from jax import numpy as jnp

from dreamerv2.distribution import log_binary_probability, log_gaussian_probability

min_denom = 0.000001


########################################################################
# Define Losses and Other Functions
########################################################################

# This returns the model loss along with intermediate agent states which are then reused to initialize the actor-critic training
def get_model_loss_and_states_function(
    config: dict,
    latent_KL: Callable,
    latent_entropy: Callable,
    networks: Dict[str, Callable],
    binary_state: bool,
    image_state: bool,
    num_actions: int,
):
    """Creates the model loss

    Returns a loss function that returns the loss and the states for
    the actor_crtic training
    """

    def model_loss_and_states(
        params: Dict[str, hk.Params],
        key: jax.random.PRNGKeyArray,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        terminals: jnp.ndarray,
    ):
        recurrent_params = params["recurrent"]
        phi_params = params["phi"]
        next_phi_params = params["next_phi"]
        reward_params = params["reward"]
        termination_params = params["termination"]

        recurrent_network = networks["recurrent"]
        phi_network = networks["phi"]
        next_phi_network = networks["next_phi"]
        reward_network = networks["reward"]
        termination_network = networks["termination"]

        if config["state_prediction_weight"] > 0.0:
            state_network = networks["state"]
            state_params = params["state"]

        # initialize hidden state for recurrent network
        h = jnp.zeros((config["num_hidden_units"]))
        loss = 0.0
        # record whether trajectory has terminated
        terminated = False
        nonterminal_steps = 0

        # generate sequential predictions
        def model_loss_loop_function(carry, data):
            h, loss, key, terminated, nonterminal_steps = carry
            observation, action, reward, terminal = data

            key, subkey = jax.random.split(key)
            phi, phi_dist = phi_network(phi_params, observation, h, subkey)

            one_hot_actions = jnp.eye(num_actions)[action]

            if config["state_prediction_weight"] > 0.0:
                S_hat_params = state_network(state_params, phi, h)
                if binary_state:
                    S_log_probs = log_binary_probability(observation, S_hat_params)
                else:
                    S_log_probs = log_gaussian_probability(observation, S_hat_params)
                log_P_S = jnp.sum(S_log_probs)
                state_prediction_loss = -log_P_S
            else:
                state_prediction_loss = 0.0

            # no need to reconstruct state on terminal steps, just need to get reward and terminal right
            state_prediction_loss = jnp.where(terminal, 0.0, state_prediction_loss)

            r_dist = reward_network(reward_params, phi, h)
            reward_loss = -log_gaussian_probability(reward, r_dist)

            gamma_dist = termination_network(termination_params, phi, h)
            termination_loss = -log_binary_probability(
                jnp.logical_not(terminal), gamma_dist
            )

            key, subkey = jax.random.split(key)
            phi_hat, phi_hat_dist = next_phi_network(next_phi_params, h, subkey)
            # KL loss applied to make current phi closer to prediction
            KL_posterior_loss = jnp.sum(
                latent_KL(phi_dist, jax.lax.stop_gradient(phi_hat_dist))
            )

            posterior_entropy_loss = -jnp.sum(latent_entropy(phi_dist))

            # KL loss applied to make prediction closer to current phi
            KL_prior_loss = jnp.sum(
                latent_KL(jax.lax.stop_gradient(phi_dist), phi_hat_dist)
            )

            step_loss = (
                config["KL_posterior_weight"] * KL_posterior_loss
                + config["KL_prior_weight"] * KL_prior_loss
                + config["posterior_entropy_weight"] * posterior_entropy_loss
                + config["reward_weight"] * reward_loss
                + config["termination_weight"] * termination_loss
                + config["state_prediction_weight"] * state_prediction_loss
            )

            # no need to predict anything that occurs after termination
            step_loss = jnp.where(terminated, 0.0, step_loss)

            loss += jnp.sum(step_loss)

            h = jnp.where(
                terminal,
                jnp.zeros((config["num_hidden_units"])),
                recurrent_network(recurrent_params, phi, one_hot_actions, h),
            )

            nonterminal_steps += jnp.logical_not(terminated)

            terminated = jnp.logical_or(terminated, terminal)
            return (h, loss, key, terminated, nonterminal_steps), (phi, h)

        (h, loss, key, terminated, nonterminal_steps), (
            phi_array,
            h_array,
        ) = jax.lax.scan(
            model_loss_loop_function,
            (h, loss, key, terminated, nonterminal_steps),
            (observations, actions, rewards, terminals),
        )
        return loss / nonterminal_steps, phi_array, h_array

    return vmap(model_loss_and_states, in_axes=(None, 0, 0, 0, 0, 0))

def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()

# takes a function which returns loss, phis, hs and transforms it to a function which returns grad(loss), phis, hs
def model_loss_and_states_to_model_grad_and_states(func, config):
    def model_loss(*args):
        return func(*args)[0]

    def _l2_loss(params, *args):
        loss = sum(
            l2_loss(w, 0.001)
            for w in jax.tree_util.tree_leaves(params)
        )
        return loss

    def model_grad_and_states(*args):
        loss, loss_grad = vgrad(lambda *args: jnp.mean(model_loss(*args)))(*args)
        
        # Add L2 regularization to obtain same results as original work
        loss_l2, loss_l2_grad = vgrad(_l2_loss)(*args)
        loss += loss_l2
        loss_grad = jax.tree_util.tree_map(lambda g1, g2: g1 + g2, loss_grad, loss_l2_grad)
        
        phi_array, h_array = func(*args)[1:]
        return loss, (
            loss_grad,
            phi_array.reshape(
                [config["sequence_length"] * config["batch_size"]]
                + list(phi_array.shape[2:])
            ),
            h_array.reshape(
                [config["sequence_length"] * config["batch_size"]]
                + list(h_array.shape[2:])
            ),
        )

    return model_grad_and_states


def get_model_eval_function(
    config,
    latent_cross_entropy,
    latent_entropy,
    networks,
    binary_state,
    image_state,
    num_actions,
):
    def model_eval(params, key, observations, actions, rewards, terminals):
        recurrent_params = params["recurrent"]
        phi_params = params["phi"]
        next_phi_params = params["next_phi"]
        reward_params = params["reward"]
        termination_params = params["termination"]

        recurrent_network = networks["recurrent"]
        phi_network = networks["phi"]
        next_phi_network = networks["next_phi"]
        reward_network = networks["reward"]
        termination_network = networks["termination"]

        if config["state_prediction_weight"] > 0.0:
            state_network = networks["state"]
            state_params = params["state"]

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
        h = jnp.zeros(config["num_hidden_units"])
        # record whether trajectory has terminated
        terminated = False

        # long tuple of things we need to maintain and update during evaluation loop, could probably clean this up.
        # Note: r_1 and r_0 predictions are useful in MinAtar in particular because rewards are almost always 1 or 0
        # thus we can observe how accurate the model is for each case
        loop_carry = (
            h,
            S_mean_logprob_tot,
            S_nonzero_tot,
            phi_mean_cross_entropy,
            phi_mean_entropy,
            r_1_count,
            r_0_count,
            r_hat_1_tot,
            r_hat_0_tot,
            gamma_1_count,
            gamma_0_count,
            gamma_hat_1_tot,
            gamma_hat_0_tot,
            key,
            terminated,
            nonterminal_steps,
        )

        def evaluate_model_loop_function(carry, data):
            (
                h,
                S_mean_logprob_tot,
                S_nonzero_tot,
                phi_mean_cross_entropy,
                phi_mean_entropy,
                r_1_count,
                r_0_count,
                r_hat_1_tot,
                r_hat_0_tot,
                gamma_1_count,
                gamma_0_count,
                gamma_hat_1_tot,
                gamma_hat_0_tot,
                key,
                terminated,
                nonterminal_steps,
            ) = carry

            observation, action, reward, terminal = data

            key, subkey = jax.random.split(key)
            phi, phi_dist = phi_network(phi_params, observation, h, subkey)

            key, subkey = jax.random.split(key)
            phi_hat, phi_hat_dist = next_phi_network(next_phi_params, h, subkey)

            one_hot_actions = jnp.eye(num_actions)[action]

            if config["state_prediction_weight"] > 0.0:
                S_hat_params = state_network(state_params, phi, h)
                if binary_state:
                    S_log_probs = log_binary_probability(observation, S_hat_params)
                else:
                    S_log_probs = log_gaussian_probability(observation, S_hat_params)

                log_P_S = jnp.mean(S_log_probs)
                S_nonzero = jnp.mean(observation)
                S_mean_logprob_tot += jnp.where(terminated, 0.0, log_P_S)
                S_nonzero_tot += jnp.where(terminated, 0.0, S_nonzero)

            r_dist = reward_network(reward_params, phi, h)
            r_hat = r_dist["mu"]

            gamma_dist = termination_network(termination_params, phi, h)
            gamma_hat = jnp.exp(log_binary_probability(1.0, gamma_dist))

            phi_mean_cross_entropy += jnp.sum(
                jnp.where(
                    terminated,
                    0.0,
                    jnp.mean(latent_cross_entropy(phi_dist, phi_hat_dist)),
                )
            )
            phi_mean_entropy += jnp.sum(
                jnp.where(terminated, 0.0, jnp.mean(latent_entropy(phi_dist)))
            )

            r_1_count += jnp.sum(jnp.where(terminated, 0.0, reward == 1.0))
            r_0_count += jnp.sum(jnp.where(terminated, 0.0, reward == 0.0))

            r_hat_1 = jnp.where(reward == 1.0, r_hat, 0.0)
            r_hat_0 = jnp.where(reward == 0.0, r_hat, 0.0)
            r_hat_1_tot += jnp.sum(jnp.where(terminated, 0.0, r_hat_1))
            r_hat_0_tot += jnp.sum(jnp.where(terminated, 0.0, r_hat_0))

            gamma_1_count += jnp.sum(
                jnp.where(terminated, 0.0, jnp.logical_not(terminal))
            )
            gamma_0_count += jnp.sum(jnp.where(terminated, 0.0, terminal))

            gamma_hat_1 = jnp.where(jnp.logical_not(terminal), gamma_hat, 0.0)
            gamma_hat_0 = jnp.where(terminal, gamma_hat, 0.0)
            gamma_hat_1_tot += jnp.sum(jnp.where(terminated, 0.0, gamma_hat_1))
            gamma_hat_0_tot += jnp.sum(jnp.where(terminated, 0.0, gamma_hat_0))

            h = jnp.where(
                terminal,
                jnp.zeros((config["num_hidden_units"])),
                recurrent_network(recurrent_params, phi, one_hot_actions, h),
            )

            nonterminal_steps += jnp.logical_not(terminated)

            terminated = jnp.logical_or(terminated, terminal)

            loop_carry = (
                h,
                S_mean_logprob_tot,
                S_nonzero_tot,
                phi_mean_cross_entropy,
                phi_mean_entropy,
                r_1_count,
                r_0_count,
                r_hat_1_tot,
                r_hat_0_tot,
                gamma_1_count,
                gamma_0_count,
                gamma_hat_1_tot,
                gamma_hat_0_tot,
                key,
                terminated,
                nonterminal_steps,
            )
            return loop_carry, None

        loop_carry = (
            h,
            S_mean_logprob_tot,
            S_nonzero_tot,
            phi_mean_cross_entropy,
            phi_mean_entropy,
            r_1_count,
            r_0_count,
            r_hat_1_tot,
            r_hat_0_tot,
            gamma_1_count,
            gamma_0_count,
            gamma_hat_1_tot,
            gamma_hat_0_tot,
            key,
            terminated,
            nonterminal_steps,
        )

        (
            h,
            S_mean_logprob_tot,
            S_nonzero_tot,
            phi_mean_cross_entropy,
            phi_mean_entropy,
            r_1_count,
            r_0_count,
            r_hat_1_tot,
            r_hat_0_tot,
            gamma_1_count,
            gamma_0_count,
            gamma_hat_1_tot,
            gamma_hat_0_tot,
            key,
            terminated,
            nonterminal_steps,
        ), _ = jax.lax.scan(
            evaluate_model_loop_function,
            loop_carry,
            (observations, actions, rewards, terminals),
        )

        metrics = {
            "gamma_0_tot": gamma_hat_0_tot,
            "gamma_1_tot": gamma_hat_1_tot,
            "r_0_tot": r_hat_0_tot,
            "r_1_tot": r_hat_1_tot,
            "gamma_0_count": gamma_0_count,
            "gamma_1_count": gamma_1_count,
            "r_0_count": r_0_count,
            "r_1_count": r_1_count,
            "phi_cross_entropy": phi_mean_cross_entropy,
            "phi_entropy": phi_mean_entropy,
            "S_logprob_tot": S_mean_logprob_tot,
            "S_nonzero_tot": S_nonzero_tot,
            "nonterminal_steps": nonterminal_steps,
        }
        return metrics

    def multi_model_eval(*args):
        metrics = vmap(model_eval, in_axes=(None, 0, 0, 0, 0, 0))(*args)
        nonterminal_steps = jnp.sum(metrics["nonterminal_steps"]) + 1e-6
        r_0_count = jnp.sum(metrics["r_0_count"])
        r_1_count = jnp.sum(metrics["r_1_count"])
        gamma_0_count = jnp.sum(metrics["gamma_0_count"])
        gamma_1_count = jnp.sum(metrics["gamma_1_count"])
        gamma_0_tot = jnp.sum(metrics["gamma_0_tot"])
        gamma_1_tot = jnp.sum(metrics["gamma_1_tot"])
        r_0_tot = jnp.sum(metrics["r_0_tot"])
        r_1_tot = jnp.sum(metrics["r_1_tot"])
        S_logprob_tot = jnp.sum(metrics["S_logprob_tot"])
        S_nonzero_tot = jnp.sum(metrics["S_nonzero_tot"])
        phi_cross_entropy = jnp.sum(metrics["phi_cross_entropy"])
        phi_entropy = jnp.sum(metrics["phi_entropy"])
        combined_metrics = {
            "gamma_0_pred": gamma_0_tot / gamma_0_count,
            "gamma_1_pred": gamma_1_tot / gamma_1_count,
            "r_0_pred": r_0_tot / r_0_count,
            "r_1_pred": r_1_tot / r_1_count,
            "gamma_0_frac": gamma_0_count / nonterminal_steps,
            "gamma_1_frac": gamma_1_count / nonterminal_steps,
            "r_0_frac": r_0_count / nonterminal_steps,
            "r_1_frac": r_1_count / nonterminal_steps,
            "phi_cross_entropy": phi_cross_entropy / nonterminal_steps,
            "phi_entropy": phi_entropy / nonterminal_steps,
            "S_logprob": S_logprob_tot / nonterminal_steps,
            "S_nonzero_tot": S_nonzero_tot / nonterminal_steps,
        }
        return combined_metrics

    return jit(multi_model_eval)


def get_AC_loss_function(
    config, actor_network, critic_network, model_networks, num_actions
):
    def AC_loss(
        actor_params, fast_critic_params, slow_critic_params, model_params, key, phi, h
    ):
        reward_params = model_params["reward"]
        recurrent_params = model_params["recurrent"]
        termination_params = model_params["termination"]
        next_phi_params = model_params["next_phi"]

        reward_network = model_networks["reward"]
        recurrent_network = model_networks["recurrent"]
        termination_network = model_networks["termination"]
        next_phi_network = model_networks["next_phi"]

        def model_trajectory_loop_function(carry, data):
            h, phi, key = carry

            # Just use mean reward when sampling
            reward_dist = reward_network(reward_params, phi, h)
            reward = reward_dist["mu"]

            gamma_dist = termination_network(termination_params, phi, h)
            gamma = (
                jnp.exp(log_binary_probability(True, gamma_dist)) * config["discount"]
            )

            curr_pi_logit = actor_network(actor_params, phi, h)
            fast_curr_V = critic_network(fast_critic_params, phi, h)
            slow_curr_V = critic_network(slow_critic_params, phi, h)

            key, subkey = jax.random.split(key)
            action = jax.random.categorical(subkey, curr_pi_logit)
            one_hot_action = jnp.eye(num_actions)[action]

            h = recurrent_network(recurrent_params, phi, one_hot_action, h)

            key, subkey = jax.random.split(key)
            phi, _ = jax.lax.stop_gradient(next_phi_network(next_phi_params, h, subkey))

            return (h, phi, key), (
                curr_pi_logit,
                fast_curr_V,
                slow_curr_V,
                one_hot_action,
                gamma,
                reward,
            )

        # gather model trajectory
        (h, phi, key), (pi_logits, f_Vs, s_Vs, actions, gammas, rewards) = jax.lax.scan(
            model_trajectory_loop_function,
            (h, phi, key),
            jnp.arange(config["rollout_length"]),
        )

        # compute final reward, gamma and value for first bootstrapped return
        reward_dist = reward_network(reward_params, phi, h)
        reward = reward_dist["mu"]

        gamma_dist = termination_network(termination_params, phi, h)
        gamma = jnp.exp(log_binary_probability(True, gamma_dist)) * config["discount"]

        slow_curr_V = critic_network(slow_critic_params, phi, h)

        def compute_loss_loop_function(carry, data):
            G, loss = carry
            pi_logit, f_V, s_V, action, gamma, reward = data

            critic_loss = jnp.mean(0.5 * (G - f_V) ** 2)
            entropy = -jnp.sum(jax.nn.log_softmax(pi_logit) * jax.nn.softmax(pi_logit))

            actor_loss = jnp.mean(
                -0.5
                * jax.lax.stop_gradient(G - s_V)
                * jnp.sum(jax.nn.log_softmax(pi_logit) * action)
                - config["beta"] * entropy
            )

            G = reward + gamma * (
                (1 - config["lmbda"]) * jax.lax.stop_gradient(s_V) + config["lmbda"] * G
            )

            loss += jnp.mean(critic_loss + actor_loss)

            return (G, loss), None

        loss = 0.0
        G = reward + gamma * jax.lax.stop_gradient(slow_curr_V)
        # process model trajectory in reverse
        (G, loss), _ = jax.lax.scan(
            compute_loss_loop_function,
            (G, loss),
            (pi_logits, f_Vs, s_Vs, actions, gammas, rewards),
            reverse=True,
        )

        return loss / config["rollout_length"]

    return lambda *args: jnp.mean(
        vmap(AC_loss, in_axes=(None, None, None, None, 0, 0, 0))(*args)
    )


def get_act_function(actor_network, recurrent_network, num_actions):
    def act(actor_params, recurrent_params, phi, h, key, action, sample_action):
        pi_logit = actor_network(actor_params, phi, h)

        if sample_action:
            key, subkey = jax.random.split(key)
            a = jax.random.categorical(subkey, pi_logit)
        else:
            a = action

        h = recurrent_network(recurrent_params, phi, jnp.eye(num_actions)[a], h)
        return a, h

    return act


def get_observe_function(phi_network):
    def observe(phi_params, observation, phi, h, key):
        key, subkey = jax.random.split(key)
        phi, _ = phi_network(phi_params, observation, h, subkey)
        return phi

    return observe

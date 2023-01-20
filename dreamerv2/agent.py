import copy

import haiku as hk
import jax
from jax import jit, grad
from jax import numpy as jnp
import optax

from dreamerv2.buffer import replay_buffer
from dreamerv2.optimizers import apply_updates
from dreamerv2.loss import (
    get_model_loss_and_states_function,
    model_loss_and_states_to_model_grad_and_states,
    get_model_eval_function,
    get_AC_loss_function,
    get_observe_function,
    get_act_function,
)
from dreamerv2.distribution import get_latent_type
from dreamerv2 import networks as nets


########################################################################
# Define Agent
########################################################################
class dreamer_agent:
    def __init__(self, key, env, config, params=None):
        self.beta = config["beta"]
        self.t = 0

        self.key = key

        self.config = config

        self.valid_actions = list(range(env.action_space.n))

        (
            self.latent_KL,
            self.latent_entropy,
            self.latent_cross_entropy,
        ) = get_latent_type(config["latent_type"])

        num_actions = len(self.valid_actions)

        state_shape = env.observation_space.sample().shape
        if env.observation_space.sample().dtype == bool:
            self.binary_state = True
        else:
            self.binary_state = False

        # Model related initialization
        # ============================

        optimizer = optax.chain(
            optax.clip(config["grad_clip"]),
            optax.adamw(
                config["model_alpha"],
                eps=config["eps_adam"],
                weight_decay=config["wd_adam"],
                eps_root=1e-5,
            ),
        )

        model_params = {}
        self.model_networks = {}

        # dummy variables for network initialization
        self.key, dummy_key = jax.random.split(self.key)
        dummy_state = jnp.zeros(list(state_shape))
        dummy_phi = jnp.zeros(
            (
                config["num_features"]
                * (
                    config["feature_width"]
                    if config["latent_type"] == "categorical"
                    else 1
                )
            )
        )
        dummy_a = jnp.zeros((num_actions))
        dummy_h = jnp.zeros((config["num_hidden_units"]))

        self.key, subkey = jax.random.split(self.key)
        self.buffer = replay_buffer(
            config["buffer_size"], state_shape, subkey, config["cpu_replay"]
        )

        # initialize recurrent network
        recurrent_net = hk.without_apply_rng(
            hk.transform(
                lambda phi, a, h: nets.recurrent_network(config)(phi, a, h)
            )
        )

        self.key, subkey = jax.random.split(self.key)
        recurrent_params = recurrent_net.init(subkey, dummy_phi, dummy_a, dummy_h)

        recurrent_apply = recurrent_net.apply
        model_params["recurrent"] = recurrent_params
        self.model_networks["recurrent"] = recurrent_apply

        # initialize phi network
        if len(state_shape) > 1 and not config["no_conv"]:
            image_state = True
            phi_net = hk.without_apply_rng(
                hk.transform(lambda s, h, k: nets.phi_conv_network(config)(s, h, k))
            )
        else:
            image_state = False
            phi_net = hk.without_apply_rng(
                hk.transform(lambda s, h, k: nets.phi_flat_network(config)(s, h, k))
            )
        phi_apply = phi_net.apply
        self.key, subkey = jax.random.split(self.key)
        phi_params = phi_net.init(subkey, dummy_state, dummy_h, dummy_key)
        model_params["phi"] = phi_params
        self.model_networks["phi"] = phi_apply

        # initialize reward network
        reward_net = hk.without_apply_rng(
            hk.transform(lambda phi, h: nets.reward_network(config)(phi, h))
        )
        self.key, subkey = jax.random.split(self.key)
        reward_params = reward_net.init(subkey, dummy_phi, dummy_h)
        reward_apply = reward_net.apply
        model_params["reward"] = reward_params
        self.model_networks["reward"] = reward_apply

        # initialize termination network
        termination_net = hk.without_apply_rng(
            hk.transform(lambda phi, h: nets.termination_network(config)(phi, h))
        )
        self.key, subkey = jax.random.split(self.key)
        termination_params = termination_net.init(subkey, dummy_phi, dummy_h)
        termination_apply = termination_net.apply
        model_params["termination"] = termination_params
        self.model_networks["termination"] = termination_apply

        # initialize phi prediction network
        next_phi_net = hk.without_apply_rng(
            hk.transform(lambda h, k: nets.next_phi_network(config)(h, k))
        )
        next_phi_apply = next_phi_net.apply
        self.key, subkey = jax.random.split(self.key)
        next_phi_params = next_phi_net.init(subkey, dummy_h, dummy_key)
        model_params["next_phi"] = next_phi_params
        self.model_networks["next_phi"] = next_phi_apply

        # initialize state reconstruction network
        if config["state_prediction_weight"] > 0.0:
            if image_state:
                state_net = hk.without_apply_rng(
                    hk.transform(
                        lambda phi, h: nets.state_conv_network(
                            config, self.binary_state, state_shape
                        )(phi, h)
                    )
                )
            else:
                state_width = 1
                for j in state_shape:
                    state_width *= j
                state_net = hk.without_apply_rng(
                    hk.transform(
                        lambda phi, h: nets.state_flat_network(
                            config, self.binary_state, state_width
                        )(phi, h)
                    )
                )
            state_apply = state_net.apply
            self.key, subkey = jax.random.split(self.key)
            state_params = state_net.init(subkey, dummy_phi, dummy_h)
            model_params["state"] = state_params
            self.model_networks["state"] = state_apply

        if params is not None:
            model_params = params["model"]
        model_opt_state = optimizer.init(model_params)

        self.model_state = {"params": model_params, "opt_state": model_opt_state}
        self.model_update = jit(lambda p, s, g: apply_updates(optimizer, p, s, g, -1))

        model_loss_and_states = get_model_loss_and_states_function(
            config,
            self.latent_KL,
            self.latent_entropy,
            self.model_networks,
            self.binary_state,
            image_state,
            num_actions,
        )

        # this returns model_grads, model_states
        self.model_grad_and_states = jit(
            model_loss_and_states_to_model_grad_and_states(
                model_loss_and_states, self.config
            )
        )

        self.model_eval = get_model_eval_function(
            self.config,
            self.latent_cross_entropy,
            self.latent_entropy,
            self.model_networks,
            self.binary_state,
            image_state,
            num_actions,
        )

        # AC related initialization
        # =========================
        actor_optimizer = optax.chain(
            optax.clip(config["grad_clip"]),
            optax.adamw(
                config["actor_alpha"],
                eps=config["eps_adam"],
                weight_decay=config["wd_adam"],
            ),
        )

        critic_optimizer = optax.chain(
            optax.clip(config["grad_clip"]),
            optax.adamw(
                config["critic_alpha"],
                eps=config["eps_adam"],
                weight_decay=config["wd_adam"],
            ),
        )

        # initialize actor network
        actor_net = hk.without_apply_rng(
            hk.transform(lambda phi, h: nets.actor_network(config, num_actions)(phi, h))
        )
        self.actor_apply = actor_net.apply
        self.key, subkey = jax.random.split(self.key)
        actor_params = actor_net.init(subkey, dummy_phi, dummy_h)

        # initialize fast critic network
        critic_net = hk.without_apply_rng(
            hk.transform(lambda phi, h: nets.critic_network(config)(phi, h))
        )
        self.critic_apply = critic_net.apply
        self.key, subkey = jax.random.split(self.key)
        fast_critic_params = critic_net.init(subkey, dummy_phi, dummy_h)

        if params is not None:
            fast_critic_params = params["critic"]
            actor_params = params["actor"]

        self.slow_critic_params = copy.deepcopy(fast_critic_params)

        actor_opt_state = actor_optimizer.init(actor_params)
        self.actor_state = {"params": actor_params, "opt_state": actor_opt_state}
        self.actor_update = jit(
            lambda p, s, g: apply_updates(actor_optimizer, p, s, g, -1)
        )

        critic_opt_state = critic_optimizer.init(fast_critic_params)
        self.critic_state = {
            "params": fast_critic_params,
            "opt_state": critic_opt_state,
        }
        self.critic_update = jit(
            lambda p, s, g: apply_updates(critic_optimizer, p, s, g, -1)
        )

        self.AC_loss_grad = jit(
            grad(
                get_AC_loss_function(
                    self.config,
                    self.actor_apply,
                    self.critic_apply,
                    self.model_networks,
                    num_actions,
                ),
                argnums=(0, 1, 2),
            )
        )

        # maintain state information for acting in the real world
        self.h = jnp.zeros(dummy_h.shape)
        self.phi = jnp.zeros(dummy_phi.shape)

        self._observe = jit(get_observe_function(self.model_networks["phi"]))
        self._act = jit(
            get_act_function(
                self.actor_apply, self.model_networks["recurrent"], num_actions
            ),
            static_argnames=("sample_action"),
        )

    def act(self, observation, random=False):
        self.key, subkey = jax.random.split(self.key)
        self.phi = self._observe(
            self.model_params()["phi"],
            jnp.asarray(observation, dtype=float),
            self.phi,
            self.h,
            subkey,
        )

        if random:
            self.key, subkey = jax.random.split(self.key)
            action = int(jax.random.choice(subkey, jnp.array(self.valid_actions)))
        else:
            action = 0

        self.key, subkey = jax.random.split(self.key)
        action, self.h = self._act(
            self.actor_params(),
            self.model_params()["recurrent"],
            # self.model_params(),
            self.phi,
            self.h,
            subkey,
            action,
            not random,
        )

        return self.valid_actions[int(action)]

    def reset(self):
        self.h = jnp.zeros(self.h.shape)
        self.phi = jnp.zeros(self.phi.shape)

    def model_params(self):
        return self.model_state["params"]

    def actor_params(self):
        return self.actor_state["params"]

    def critic_params(self):
        return self.critic_state["params"]

    def add_to_replay(self, *args):
        self.buffer.add(*args)

    def get_buffers(self):
        return self.buffer.get_buffers()

    def set_buffers(self, buffers):
        self.buffer.set_buffers(buffers)

    def update(self):
        observations, actions, rewards, terminals = self.buffer.sample(
            self.config["batch_size"], self.config["sequence_length"]
        )
        self.key, subkey = jax.random.split(self.key)
        subkeys = jax.random.split(subkey, num=self.config["batch_size"])

        loss, (grads, phis, hs) = self.model_grad_and_states(
            self.model_params(), subkeys, observations, actions, rewards, terminals
        )
        # print("loss = ", loss)
        # input()
        # # is_nan_in_grads = jnp.any(jnp.isnan(jax.tree_util.tree_flatten(grads)))
        # nan_in_grads = jax.tree_util.tree_map(lambda g: jnp.isnan(g), grads)
        # print("nan_in_grads = ", nan_in_grads["recurrent"])

        self.model_state["params"], self.model_state["opt_state"] = self.model_update(
            self.model_state["params"], self.model_state["opt_state"], grads
        )

        self.key, subkey = jax.random.split(self.key)
        subkeys = jax.random.split(
            subkey, num=self.config["batch_size"] * self.config["sequence_length"]
        )
        grads = self.AC_loss_grad(
            self.actor_params(),
            self.critic_params(),
            self.slow_critic_params,
            self.model_params(),
            subkeys,
            phis,
            hs,
        )
        self.actor_state["params"], self.actor_state["opt_state"] = self.actor_update(
            self.actor_params(), self.actor_state["opt_state"], grads[0]
        )
        (
            self.critic_state["params"],
            self.critic_state["opt_state"],
        ) = self.critic_update(
            self.critic_params(), self.critic_state["opt_state"], grads[1]
        )

        self.t += 1

    def current_value(self, observation):
        self.key, subkey = jax.random.split(self.key)
        phi = self._observe(
            # self.model_params()["phi"],
            self.model_params(),
            jnp.asarray(observation, dtype=float),
            self.phi,
            self.h,
            subkey,
        )
        return self.critic_apply(self.critic_params(), phi, self.h)

    def sync_slow_critic(self):
        self.slow_critic_params = copy.deepcopy(self.critic_params())

    def eval(self):
        observations, actions, rewards, terminals = self.buffer.sample(
            self.config["batch_size"], self.config["sequence_length"]
        )
        self.key, subkey = jax.random.split(self.key)
        subkeys = jax.random.split(subkey, num=self.config["batch_size"])
        return self.model_eval(
            self.model_params(), subkeys, observations, actions, rewards, terminals
        )

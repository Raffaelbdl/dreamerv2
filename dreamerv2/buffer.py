import jax
from jax import jit, vmap
from jax import numpy as jnp
import numpy as np


########################################################################
# Define Replay Buffer
########################################################################

# draws a sequence of length sequence length from each data buffer begining at i, except shifted to extend backward as much as possible to maximize sequence length within episode
def draw_sequence(
    i,
    sequence_length,
    max_index,
    buffer_end_location,
    observations,
    actions,
    rewards,
    terminals,
):
    # draw terminal sequence, wrapping at end of replay buffer
    terms = jnp.take(terminals, jnp.arange(sequence_length) + i, axis=0, mode="wrap")
    # find first terminal in sequence (if any)
    first_terminal_index = jnp.nonzero(terms, size=1, fill_value=-1)[0]

    # shift i such that the sequence ends at the first terminal index (if any are present)
    i = jnp.where(
        first_terminal_index != -1,
        (i - (sequence_length - 1 - first_terminal_index)) % max_index,
        i,
    )

    # draw terminals in new sequence
    terms = jnp.take(terminals, jnp.arange(sequence_length) + i, axis=0, mode="wrap")
    # find last terminal in sequence besides the one found in the last step (if any others are present)
    last_terminal_index = jnp.nonzero(jnp.flip(terms[:-1]), size=1, fill_value=-1)[0]

    # shift i to just after the second last terminal sequence (only if we found a terminal in the first step and another in the second)
    i = jnp.where(
        jnp.logical_and(first_terminal_index != -1, last_terminal_index != -1),
        (i + sequence_length - last_terminal_index - 1) % max_index,
        i,
    )

    end_index = i + sequence_length

    # check if the end of the buffer is included in the interval (note this may result in the occasional shorter sequence but this shouldn't matter much)
    buffer_end_in_interval = jnp.where(
        end_index <= max_index,
        jnp.logical_and(i <= buffer_end_location, buffer_end_location < end_index),
        jnp.logical_or(
            buffer_end_location < end_index - max_index, buffer_end_location >= i
        ),
    )

    # shift sequences so they do not include the end of the buffer
    i = jnp.where(
        buffer_end_in_interval, (buffer_end_location - sequence_length) % max_index, i
    )

    # sample other indices
    obs = jnp.take(observations, jnp.arange(sequence_length) + i, axis=0, mode="wrap")
    acts = jnp.take(actions, jnp.arange(sequence_length) + i, axis=0, mode="wrap")
    rs = jnp.take(rewards, jnp.arange(sequence_length) + i, axis=0, mode="wrap")
    terms = jnp.take(terminals, jnp.arange(sequence_length) + i, axis=0, mode="wrap")

    return obs, acts, rs, terms


# transform single sequence draw into batched versoin, only apply vmap to input indices
draw_sequences = jit(
    vmap(draw_sequence, in_axes=(0, None, None, None, None, None, None, None)),
    static_argnums=(1),
)

# numpy implementation of draw_sequences for case where replay buffer must be stored in RAM, this could be better optimized
def draw_sequences_np(
    indices,
    sequence_length,
    max_index,
    buffer_end_location,
    observations,
    actions,
    rewards,
    terminals,
):
    all_obs = []
    all_acts = []
    all_rs = []
    all_terms = []
    # just sample indices sequentially here
    for i in indices:
        # draw terminal sequence, wrapping at end of replay buffer
        terms = np.take(terminals, np.arange(sequence_length) + i, axis=0, mode="wrap")
        # find first terminal in sequence (if any)
        terminal_indices = np.nonzero(terms)[0]
        first_terminal_index = -1 if len(terminal_indices) == 0 else terminal_indices[0]

        # shift i such that the sequence ends at the first terminal index (if any are present)
        i = np.where(
            first_terminal_index != -1,
            (i - (sequence_length - 1 - first_terminal_index)) % max_index,
            i,
        )

        # draw terminals in new sequence
        terms = np.take(terminals, np.arange(sequence_length) + i, axis=0, mode="wrap")
        # find last terminal in sequence besides the one found in the last step (if any others are present)
        other_terminal_indices = np.nonzero(np.flip(terms[:-1]))[0]
        last_terminal_index = (
            -1 if len(other_terminal_indices) == 0 else other_terminal_indices[0]
        )

        # shift i to just after the second last terminal sequence (only if we found a terminal in the first step and another in the second)
        i = np.where(
            np.logical_and(first_terminal_index != -1, last_terminal_index != -1),
            (i + sequence_length - last_terminal_index - 1) % max_index,
            i,
        )

        end_index = i + sequence_length

        # check if the end of the buffer is included in the interval (note this may result in the occasional shorter sequence but this shouldn't matter much)
        buffer_end_in_interval = np.where(
            end_index <= max_index,
            np.logical_and(i <= buffer_end_location, buffer_end_location < end_index),
            np.logical_or(
                buffer_end_location < end_index - max_index, buffer_end_location >= i
            ),
        )

        # shift sequences so they do not include the end of the buffer
        i = np.where(
            buffer_end_in_interval,
            (buffer_end_location - sequence_length) % max_index,
            i,
        )

        # sample other indices
        all_obs += [
            np.take(observations, np.arange(sequence_length) + i, axis=0, mode="wrap")
        ]
        all_acts += [
            np.take(actions, np.arange(sequence_length) + i, axis=0, mode="wrap")
        ]
        all_rs += [
            np.take(rewards, np.arange(sequence_length) + i, axis=0, mode="wrap")
        ]
        all_terms += [
            np.take(terminals, np.arange(sequence_length) + i, axis=0, mode="wrap")
        ]
    return np.stack(all_obs), np.stack(all_acts), np.stack(all_rs), np.stack(all_terms)


class replay_buffer:
    def __init__(self, buffer_size, obs_shape, key, use_cpu=False):
        self.key = key
        self.buffer_size = buffer_size
        self.location = 0
        self.full = False
        self.use_cpu = use_cpu

        # if use_cpu is set replay buffer uses ordinary numpy arrays, otherwise jax.numpy arrays
        if self.use_cpu:
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
        if self.use_cpu:
            self.observations[self.location] = np.asarray(obs, dtype=float)
            self.actions[self.location] = np.asarray(action, dtype=int)
            self.rewards[self.location] = np.asarray(reward, dtype=float)
            self.terminals[self.location] = np.asarray(terminal, dtype=bool)
        else:
            self.observations = self.observations.at[self.location].set(
                jnp.asarray(obs, dtype=float)
            )
            self.actions = self.actions.at[self.location].set(
                jnp.asarray(action, dtype=int)
            )
            self.rewards = self.rewards.at[self.location].set(
                jnp.asarray(reward, dtype=float)
            )
            self.terminals = self.terminals.at[self.location].set(
                jnp.asarray(terminal, dtype=bool)
            )
        if self.location == self.buffer_size - 1:
            self.full = True
        # increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size, sequence_length):
        max_index = self.buffer_size if self.full else self.location

        self.key, subkey = jax.random.split(self.key)
        start_indices = jax.random.choice(subkey, max_index, shape=(batch_size,))
        if self.use_cpu:
            return draw_sequences_np(
                start_indices,
                sequence_length,
                max_index,
                self.location,
                self.observations,
                self.actions,
                self.rewards,
                self.terminals,
            )
        else:
            return draw_sequences(
                start_indices,
                sequence_length,
                max_index,
                self.location,
                self.observations,
                self.actions,
                self.rewards,
                self.terminals,
            )

    # useful for pickling to a file
    def get_buffers(self):
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminals": self.terminals,
        }

    def set_buffers(self, buffers):
        # assume that buffers being set are full since we have no way to tell apriori
        self.observations = buffers["observations"]
        self.actions = buffers["actions"]
        self.rewards = buffers["rewards"]
        self.terminals = buffers["terminals"]
        self.location = 0
        self.full = True
        self.buffer_size = self.observations.shape[0]

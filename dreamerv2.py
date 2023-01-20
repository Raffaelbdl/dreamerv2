import json
import argparse
import time
import pickle as pkl
import gym
import jax
from jax.config import config
import jax.numpy as jnp

config.update("jax_debug_nans", False)
# config.update("jax_enable_x64", True)

from dreamerv2.agent import dreamer_agent


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
    parser.add_argument("--config", "-c", type=str, default="config.json")
    parser.add_argument("--wandb", "-w", type=bool, default=False)
    args = parser.parse_args()
    key = jax.random.PRNGKey(args.seed)

    if args.load is not None:
        with open(args.load, "rb") as f:
            params = pkl.load(f)
            print("loaded model params from " + args.load)
    else:
        params = None

    with open(args.config, "r") as f:
        config = json.load(f)

    def get_config(k, d, default=None):
        if k not in d:
            d[k] = default
        return d[k]

    eval_frequency = get_config("eval_frequency", config, None)
    save_frequency = get_config("save_frequency", config, None)
    slow_critic_interval = get_config("slow_critic_interval", config, None)
    num_frames = get_config("num_frames", config, None)
    training_start_time = get_config("training_start_time", config, None)
    train_frequency = get_config("train_frequency", config, None)

    ########################################################################
    # Initialize Agent and Environments
    ########################################################################
    env = gym.make(config["env"])
    key, subkey = jax.random.split(key)
    env.seed(int(subkey[0]))

    key, subkey = jax.random.split(key)
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
        if isinstance(state, tuple): # handle new gym outputs
            state = state[0]

        reward = 0.0
        terminal = False
        while (not terminal) and t < num_frames:
            # act randomly until training begins
            action = agent.act(state.astype(float), random=(t <= training_start_time))

            agent.add_to_replay(state, action, reward, terminal)

            step = env.step(action)
            if len(step) == 4:
                state, reward, terminal, _ = step
            elif len(step) == 5:
                state, reward, terminal, _, _ = step
            else:
                raise ValueError()

            curr_return += reward

            if terminal:
                episode += 1
                termination_times += [t]
                avg_return = 0.99 * avg_return + 0.01 * curr_return
                returns += [curr_return]
                curr_return = 0.0

                # add terminal experience to replay immediately, action should be irrelevant since we don't take action in terminal states
                agent.add_to_replay(state, action, reward, terminal)

            if (t >= training_start_time) and (t % train_frequency == 0):
                agent.update()

            if (t >= training_start_time) and (t % slow_critic_interval == 0):
                agent.sync_slow_critic()
            t += 1

            ########################################################################
            # Logging
            ########################################################################
            if t % eval_frequency == 0:
                metrics = agent.eval()
                print(
                    "Avg return: "
                    + str(
                        0.0
                        if episode == 0
                        else jnp.around(avg_return / (1 - 0.99**episode), 2)
                    )
                )
                print("Frame: " + str(t))
                print("Time per frame: " + str((time.time() - t_start) / t))
                for i, k in enumerate(metrics):
                    print(
                        (k + ": " + str(metrics[k])).ljust(30),
                        end=("\n" if i > 1 and (i + 1) % 4 == 0 else "| "),
                    )
                    if k in evaluation_metrics:
                        evaluation_metrics[k] += [metrics[k]]
                    else:
                        evaluation_metrics[k] = [metrics[k]]
                print()


            # dump data every save-frequency frames
            if t % save_frequency == 0:
                # data is always dumbed to the same file and overwritten
                with open(args.output, "wb") as f:
                    pkl.dump(
                        {
                            "config": config,
                            "returns": returns,
                            "termination_times": termination_times,
                            "evaluation_metrics": evaluation_metrics,
                        },
                        f,
                    )

                # model is saved seperately at each save time
                with open(args.model + "_" + str(t), "wb") as f:
                    pkl.dump(
                        {
                            "model": agent.model_params(),
                            "actor": agent.actor_params(),
                            "critic": agent.critic_params(),
                        },
                        f,
                    )

    ########################################################################
    # Save Data and Weights
    ########################################################################
    with open(args.output, "wb") as f:
        pkl.dump(
            {
                "config": config,
                "returns": returns,
                "termination_times": termination_times,
                "evaluation_metrics": evaluation_metrics,
            },
            f,
        )

    with open(args.model, "wb") as f:
        pkl.dump(
            {
                "model": agent.model_params(),
                "actor": agent.actor_params(),
                "critic": agent.critic_params(),
            },
            f,
        )

    with open(args.buffer, "wb") as f:
        pkl.dump(agent.get_buffers(), f)

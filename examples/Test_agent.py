"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""
import json
import os

import torch.nn
import numpy as np

from rlpyt.envs.gym import make as gym_make
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.utils.buffer import torchify_buffer, numpify_buffer
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.collections import namedarraytuple

# TODO: Importing SamplesToBufferTl here is done to avoid problems when loading expert demos. This can be avoided by switching to namedarraytupleschema in the creation of thr replayBuffer

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
                                    SamplesToBuffer._fields + ("timeout",))

def get_saved_session_path(configuration_dict):
    log_dir = configuration_dict['logger']['log_dir']
    run_ID = configuration_dict['logger']['run_ID']
    log_dir = os.path.join(log_dir, f"run_{run_ID}")
    exp_dir = os.path.abspath(log_dir)
    saved_session_path = os.path.join(exp_dir, 'params.pkl')
    return saved_session_path

def evaluate_agent(configuration_dict, episodes, display=True):
    env = gym_make(configuration_dict['sampler']['eval_env_kwargs']['id'])
    _ = env.reset()
    saved_session_path = get_saved_session_path(configuration_dict)
    if os.path.isfile(saved_session_path):
        try:
            data = torch.load(saved_session_path)
            itr = data['itr']  # might be useful for logging / debugging
            cum_steps = data['cum_steps']  # might be useful for logging / debugging
            agent_state_dict = data['agent_state_dict']  # 'model' and 'target' keys
            optimizer_state_dict = data['optimizer_state_dict']
            agent = SacAgent(initial_model_state_dict=agent_state_dict, **configuration_dict['agent'])
            agent.initialize(env.spaces, share_memory=False,
                             global_B=1, env_ranks=[0])
        except:
            raise FileNotFoundError(f'Couldn\'t find a pretrained agent in saved_session_path')
    else:
        raise FileNotFoundError(f'Couldn\'t find a pretrained agent in saved_session_path')

    tot_rewards = []
    for i in range(episodes):
        _ = env.reset()
        a = env.action_space.sample()
        o, r, d, env_info = env.step(a)
        r = np.asarray(r, dtype="float32")  # Must match torch float dtype here.
        agent.reset()
        done = False
        tot_reward = 0
        t = 0
        while not done:
            agent_inputs = torchify_buffer(AgentInputs(o, a, np.array([r])))
            action, agent_info = agent.step(*agent_inputs)
            action = numpify_buffer(action)
            o, r, done, info = env.step(action)
            env.render()
            tot_reward += r
            t += 1
        tot_rewards.append(tot_reward)
        print(f'Total reward for episode {i}: {tot_reward}')
    print(f'Average reward = {np.mean(tot_rewards)}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--configuration_path', help='Path to load/save configuration file', type=str)
    # /home/sagiv/Documents/HUJI/Tsevi/RL/rlpyt/data/local/20211005/152843/SACfD_ablation/SACfD_ablation_ER_0.2/run_1/conf.json
    args = parser.parse_args()
    if args.configuration_path is not None and \
            args.configuration_path.lower().endswith('.json') and \
            os.path.isfile(args.configuration_path):
        with open(args.configuration_path, 'r+') as conf_file:
            config = json.loads(conf_file.read())
        evaluate_agent(config, 10, True)

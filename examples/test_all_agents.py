import json
import os

import torch.nn
import numpy as np
import pandas as pd

from rlpyt.envs.gym import make as gym_make
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.utils.buffer import torchify_buffer, numpify_buffer
from rlpyt.agents.base import AgentInputs
import os
import numpy as np
import json
import csv
import time
import multiprocessing as mp

from rlpyt.utils.collections import namedarraytuple

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
                                    SamplesToBuffer._fields + ("timeout",))


def list_files(filepath, filetype):
    paths = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.lower().endswith(filetype.lower()):
                paths.append(os.path.join(root, file))
    return (paths)

def get_saved_session_path(configuration_dict):
    log_dir = configuration_dict['logger']['log_dir']
    run_ID = configuration_dict['logger']['run_ID']
    log_dir = os.path.join(log_dir, f"run_{run_ID}")
    exp_dir = os.path.abspath(log_dir)
    saved_session_path = os.path.join(exp_dir, 'params.pkl')
    return saved_session_path

def run_agent(env,agent,episodes,seeds=None,display=False):
    tot_rewards = []
    for i in range(episodes):
        if seeds is not None:
            env.seed(int(seeds[i]))
        initials = env.reset()
        a = env.action_space.sample()
        o, r, d, env_info = env.step(a)
        r = np.asarray(r, dtype="float32")  # Must match torch float dtype here.
        agent.reset()
        done = False
        tot_reward = 0
        t = 0
        while not done and t<1000:
            agent_inputs = torchify_buffer(AgentInputs(o, a, np.array([r])))
            action, agent_info = agent.step(*agent_inputs)
            action = numpify_buffer(action)
            o, r, done, info = env.step(action)
            if display:
                env.render()
            tot_reward += r
            t += 1
        tot_rewards.append(tot_reward)
        if display:
            print(f'Total reward for episode {i}: {tot_reward}')
        elif i%10 ==0:
            print(i)
    return tot_rewards,initials

def evaluate_agent(configuration_dict, episodes, display=True, seed = None,res_dict = None,initials = []):
    env = gym_make(configuration_dict['sampler']['eval_env_kwargs']['id'])
    if seed is not None:
        np.random.seed(seed)
        seeds = np.random.randint(1,9999,episodes)
    obs_size = len(env.reset())
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

    tot_rewards,this_initials = run_agent(env,agent,episodes,seeds,display)
    mar = np.mean(tot_rewards)
    if display:
        print(f'Average reward = {mar}')
    if res_dict is not None:
        log_dir = configuration_dict['logger']['log_dir']
        run_id =  configuration_dict['logger']['run_ID']
        res_dict[(log_dir.split('/')[-1],run_id)] = tot_rewards
        initials.append(this_initials)
        # print(res_dict)
    else:
        return mar,tot_rewards, this_initials

if __name__ == "__main__":
    multi = True
    seed = 1234
    episodes = 10
    # conf_paths = list_files('../data/local/', 'conf.json') #/home/sagiv/Documents/HUJI/Tsevi/RL/rlpyt/data/local/20211010/112947/SAC
    conf_paths = list_files('/home/sagiv/Documents/HUJI/Tsevi/RL/rlpyt/data/local/20211010/112947/SAC',
                            'conf.json')  #
    # conf_paths = list_files('/home/sagiv/Documents/HUJI/Tsevi/RL/rlpyt/data/local/20211005/152843/SACfD_ablation',
    #                         'conf.json')  #

    first_initials = None
    mars = np.zeros(len(conf_paths))
    rewards = np.zeros((len(conf_paths),episodes))
    avg_time = np.zeros(len(conf_paths))
    start_time = time.time()
    processes = []
    if multi:
        mgr = mp.Manager()
        res_dict = mgr.dict()
        # first_initials = mgr.list()
    for i, conf_path in enumerate(conf_paths):
        with open(conf_path, 'r+') as conf_file:
            config = json.loads(conf_file.read())
        if multi:
            log_dir = config['logger']['log_dir']
            mkwargs = {
                'configuration_dict':config,
                'episodes':episodes,
                'display': False,
                'seed' : seed,
                'res_dict' : res_dict,
                # 'initials' : first_initials
            }
            processes.append(mp.Process(target = evaluate_agent,kwargs=mkwargs))
        else:
            conf_start_time = time.time()
            print(f'Running agent {i+1} of {len(conf_paths)}')
            mar, tot_rewards, initials = evaluate_agent(config, episodes, display=False, seed=seed)
            mars[i] = mar
            rewards[i,:] = tot_rewards
            now = time.time()
            avg_time[i] = (now - conf_start_time) / episodes
            avg_time_overall = (now - start_time) / (episodes * (i + 1))
            print(f"{conf_path}\t{mar}")
            print(f"avg episode time for agent:{avg_time[i]}")
            print(f"avg episode time overall:{avg_time_overall}")
            with open('../data/conf_mars.csv', 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows(list(zip(conf_paths, mars.tolist(),avg_time.tolist())))
            np.savetxt('../data/rewards1.csv',rewards)
    if multi:
        [p.start() for p in processes]
        [p.join() for p in processes]
    print(res_dict)
    df = pd.DataFrame.from_dict({k:v for k,v in res_dict.items()})
    # print(first_initials)
    df.to_csv('../data/rewards_2.csv')

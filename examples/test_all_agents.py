import argparse
import datetime
import json
import os

import torch.nn
import numpy as np

from rlpyt.envs.gym import make as gym_make
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.utils.buffer import torchify_buffer, numpify_buffer
from rlpyt.agents.base import AgentInputs
from mpi4py import MPI
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
    print('running agent')
    tot_rewards = []
    for i in range(episodes):
        if seeds is not None:
            env.seed(int(seeds[i]))
        initials = env.reset()
        if i == 0:
            print('env reset')
        a = env.action_space.sample()
        o, r, d, env_info = env.step(a)
        r = np.asarray(r, dtype="float32")  # Must match torch float dtype here.
        agent.reset()
        if i == 0:
             print('agent reset')
        done = False
        tot_reward = 0
        t = 0
        while not done and t < 1000:
            agent_inputs = torchify_buffer(AgentInputs(o, a, np.array([r])))
            action, agent_info = agent.step(*agent_inputs)
            if i == 0:
                print('completed first step')
            action = numpify_buffer(action)
            o, r, done, info = env.step(action)
            if display:
                env.render()
            tot_reward += r
            t += 1
        tot_rewards.append(tot_reward)
        if display:
            print(f'Total reward for episode {i}: {tot_reward}')
        elif i % 10 == 0:
            print(i)
    return tot_rewards, initials


def evaluate_agent(configuration_dict, episodes, display=True, seed=None, res_dict=None, initials=[]):
    env = gym_make(configuration_dict['sampler']['eval_env_kwargs']['id'])
    if seed is not None:
        print(f'seeding')
        rng = np.random.default_rng(seed)
        seeds = rng.integers(1,9999, episodes)
        print(f'seed successful')
    obs_size = len(env.reset())
    print('env created and reset')
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
    print('agent initialized')

    tot_rewards,this_initials = run_agent(env,agent,episodes,seeds,display)
    print('agent ran')
    mar = np.mean(tot_rewards)
    if display:
        print(f'Average reward = {mar}')
    if res_dict is not None:
        log_dir = configuration_dict['logger']['log_dir']
        run_id = configuration_dict['logger']['run_ID']
        res_dict[(log_dir.split('/')[-1], run_id)] = tot_rewards
        initials.append(this_initials)
        # print(res_dict)
    else:
        print('returning results')
        return mar,tot_rewards, this_initials

if __name__ == "__main__":
    # Initialize the Parser
    parser = argparse.ArgumentParser()
    # Adding Arguments
    parser.add_argument('--folder', '-f',
                        type=str, help='Folder to search agents in', default='../data/')
    parser.add_argument('--episodes', '-e', type=int,
                        help='number of episodes to test each agent on', default=10)
    parser.add_argument('--seed', '-s', type=int,
                        help='Random seed for environments', default = 1234)
    parser.add_argument('--n_cpu', '-n', type=int,
                        help='number of available cpu cores (for parallelization)',default=1)
    args = parser.parse_args()
    args = parser.parse_args()
    folder = args.folder
    multi = not args.n_cpu in [0, 1]
    comm = MPI.COMM_WORLD
    n_process = comm.Get_size()
    mpi =  n_process >1
    seed = args.seed
    episodes = args.episodes
    conf_paths = list_files(folder, 'conf.json')  #
    first_initials = None
    mars = np.zeros(len(conf_paths))
    rewards = np.zeros((len(conf_paths), episodes))
    avg_time = np.zeros(len(conf_paths))
    start_time = time.time()
    processes = []
    map_vals = []
    for i, conf_path in enumerate(conf_paths):
        with open(conf_path, 'r+') as conf_file:
            config = json.loads(conf_file.read())
        if multi:
            log_dir = config['logger']['log_dir']
            map_vals.append([config, episodes, False, seed])
            # processes.append(mp.Process(target = evaluate_agent,kwargs=mkwargs))
        else:
            print('Running in sequential mode')
            conf_start_time = time.time()
            print(f'Running agent {i + 1} of {len(conf_paths)}')
            mar, tot_rewards, initials = evaluate_agent(config, episodes, display=False, seed=seed)
            mars[i] = mar
            rewards[i, :] = tot_rewards
            now = time.time()
            avg_time[i] = (now - conf_start_time) / episodes
            avg_time_overall = (now - start_time) / (episodes * (i + 1))
            print(f"{conf_path}\t{mar}")
            print(f"avg episode time for agent:{avg_time[i]}")
            print(f"avg episode time overall:{avg_time_overall}")
            with open('../data/conf_mars.csv', 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows(list(zip(conf_paths, mars.tolist(), avg_time.tolist())))
            np.savetxt('../data/rewards1.csv', rewards)
    if multi:
        print('Runing in parallel mode')
        if mpi:
            print('Parallel MPI mode')
            from mpi4py.futures import MPIPoolExecutor
            with MPIPoolExecutor(max_workers=args.n_cpu) as executor:
                res = executor.starmap(evaluate_agent, map_vals)
        else:
            print('Parallel Multiprocessing mode')
            with mp.Pool(args.n_cpu) as pool:
                res = pool.starmap(evaluate_agent, map_vals)
        mars = []
        tmp = np.array([v for _, _, v in res])
        ## Make sure all initial conditions are the same
        assert all([np.allclose(tmp[i, :], tmp[i + 1, :]) for i in range(tmp.shape[0] - 1)])
        tot_rewards = np.array([v for _, v, _ in res])

    index = ['__'.join(cp.split('/')[-3:-1]) for cp in conf_paths]
    with open('../data/rewards_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(index)
        writer.writerows(tot_rewards.T.tolist())

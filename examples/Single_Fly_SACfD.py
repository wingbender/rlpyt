
import json
import os
import sys

import torch.nn
import numpy as np

from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.sac import SAC
from rlpyt.algos.qpg.sacfd import SACfD
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.buffer import torchify_buffer, numpify_buffer
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel
from rlpyt.replays.non_sequence.uniform import (UniformReplayBuffer,
                                                AsyncUniformReplayBuffer)
from rlpyt.replays.non_sequence.time_limit import (TlUniformReplayBuffer,
                                                   AsyncTlUniformReplayBuffer)
import pickle
import psutil
# import gym_flySim

# TODO: Importing SamplesToBufferTl here is done to avoid problems when loading expert demos. This can be avoided by switching to namedarraytupleschema in the creation of thr replayBuffer

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
                                    SamplesToBuffer._fields + ("timeout",))


# def build_and_train(configuration_path):
def build_and_train(default_config,config_path=None,slot_affinity_code=None, log_dir=None, run_ID=None):
    if config_path is not None:
        with open(config_path,'r') as f:
            variant = json.loads(f.read())
            config = update_config(default_config, variant)
    if slot_affinity_code is not None:
        affinity = affinity_from_code(slot_affinity_code)
    else:
        pp = psutil.Process()
        cpus = pp.cpu_affinity()
        cpus = cpus[:config['sampler']['batch_B']]
        affinity = dict(cuda_idx=None, workers_cpus=list([cpu] for cpu in cpus))
    if log_dir is not None:
        variant = load_variant(log_dir)
        config = update_config(default_config, variant)
    else:
        config = default_config
    print('Starting new session')
    agent = SacAgent(**config['agent'])
    if config['general']['algo'] == 'SAC':
        del config['algo']['expert_ratio']
        del config['algo']['expert_discount']
        del config['algo']['demonstrations_path']
        algo = SAC(**config['algo'])
    elif config['general']['algo'] == 'SACfD':
        algo = SACfD(**config['algo'])
        del config['sampler']['max_decorrelation_steps']
    if config['general']['sampler_type'] == 'SerialSampler':
        sampler = SerialSampler(
            EnvCls=gym_make, **config['sampler']
        )
    elif config['general']['sampler_type'] == 'CpuSampler':
        sampler = CpuSampler(
            EnvCls=gym_make, **config['sampler']
        )
    else:
        return NotImplementedError()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config['runner'],
    )

    # name = "a2c_" + config["env"]["game"]
    # log_dir = "example_6"
    # with logger_context(log_dir, run_ID, name, config):
    #     runner.train()
    if run_ID is not None:
        config['logger']['run_ID'] = run_ID
    if log_dir is not None:
        config['logger']['log_dir'] = log_dir
    with logger_context(**config['logger']):
        exp_dir = logger.get_snapshot_dir()
        conf_filename = os.path.join(exp_dir, 'conf.json')
        with open(conf_filename, 'w') as f:
            f.writelines(json.dumps(config,indent=4))
        runner.train()

    # def show_agent(env_id, episodes, agent=None, save_name=None, run_ID=0):
def get_saved_session_path(configuration_dict):
    log_dir = configuration_dict['logger']['log_dir']
    run_ID = configuration_dict['logger']['run_ID']
    log_dir = os.path.join(log_dir, f"run_{run_ID}")
    exp_dir = os.path.abspath(log_dir)
    saved_session_path = os.path.join(exp_dir, 'params.pkl')
    return saved_session_path

if __name__ == "__main__":
    default_configuration = {
        'general':
            {
                # 'sampler_type': 'SerialSampler',  # CpuSampler
                'sampler_type': 'CpuSampler',  # CpuSampler
                'algo':'SACfD'
            },
        'agent':
            {
                'model_kwargs':
                    {
                        'hidden_sizes': [64, 64, 32, 32]
                    },  # Pi model.
                'q_model_kwargs':
                    {
                        'hidden_sizes': [64, 64]
                    },
                'v_model_kwargs':
                    {
                        'hidden_sizes': [32, 32]
                    },
            },
        'algo':
            {
                'replay_size': 10000,
                'replay_ratio': 4,
                'batch_size': 256,
                'min_steps_learn': 256,
                'demonstrations_path': '/home/sagiv/Documents/HUJI/Tsevi/RL/rlpyt/data/fly_demo.pkl',
                'expert_ratio': 0.4,
                'expert_discount':0.6
            },
        'sampler':
            {
                'env_kwargs':
                    {
                        'id': 'gym_flySim:flySim-v0',
                        'random_start': True
                    },
                'eval_env_kwargs':
                    {
                        'id': 'gym_flySim:flySim-v0',
                        'random_start': True
                    },
                'max_decorrelation_steps': 0,  # Random sampling an action to bootstrap
                'eval_max_steps': 880,
                'eval_max_trajectories': 4,

                'batch_T': 128,  # Environment steps per worker in batch
                'batch_B': 8,  # Total environments and agents
                'eval_n_envs': 4,
            },
        'runner':
            {
                'n_steps': 100000,  # Total environment steps
                'log_interval_steps': 1000,
            },
        'logger':
            {
                'log_dir': '/home/sagiv/Documents/HUJI/Tsevi/RL/rlpyt/data/flySim/',
                'run_ID': 111,
                'name': 'fly_SAC',
                'snapshot_mode': 'last',
                'use_summary_writer': True
            }
    }
    build_and_train(default_configuration,None,*sys.argv[1:])

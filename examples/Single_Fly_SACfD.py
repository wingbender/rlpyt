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
import gym_flySim

# TODO: Importing SamplesToBufferTl here is done to avoid problems when loading expert demos. This can be avoided by switching to namedarraytupleschema in the creation of thr replayBuffer

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
                                    SamplesToBuffer._fields + ("timeout",))


# def build_and_train(configuration_path):
def build_and_train(default_config,slot_affinity_code=None, log_dir=None, run_ID=None):
    if slot_affinity_code is not None:
        affinity = affinity_from_code(slot_affinity_code)
    else:
        affinity = dict(cuda_idx=None, workers_cpus=[1])
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
            f.writelines(json.dumps(config))
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
                'sampler_type': 'SerialSampler',  # CpuSampler
                # 'sampler_type': 'CpuSampler',  # CpuSampler
                'algo':'SAC'
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
                'replay_size': 5e5,
                'replay_ratio': 128,
                'batch_size': 256,
                'min_steps_learn': 256,
                'demonstrations_path': '/home/sagiv/Documents/HUJI/Tsevi/RL/rlpyt/data/lunar_demo.pkl',
                'expert_ratio': 0.2,
                'expert_discount':0.8
            },
        'sampler':
            {
                'env_kwargs':
                    {
                        'id': 'flySim-v0'
                    },
                'eval_env_kwargs':
                    {
                        'id': 'flySim-v0'
                    },
                'max_decorrelation_steps': 5000,  # Random sampling an action to bootstrap
                'eval_max_steps': 5000,
                'eval_max_trajectories': 10,

                'batch_T': 512,  # Environment steps per worker in batch
                'batch_B': 4,  # Total environments and agents
                'eval_n_envs': 2,
            },
        'runner':
            {
                'n_steps': 100000,  # Total environment steps
                'log_interval_steps': 5000,
            },
        'logger':
            {
                'log_dir': '/home/sagiv/Documents/HUJI/Tsevi/RL/rlpyt/data/flySim/',
                'run_ID': 111,
                'name': 'SAC',
                'snapshot_mode': 'last',
                'use_summary_writer': True
            }
    }
    build_and_train(default_configuration,*sys.argv[1:])

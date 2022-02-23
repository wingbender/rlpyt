
"""
Runs multiple instances of the Atari environment and optimizes using A2C
algorithm. Can choose between configurations for use of CPU/GPU for sampling
(serial or parallel) and optimization (serial).

Alternating sampler is another option.  For recurrent agents, a different mixin
is required for alternating sampling (see rlpyt.agents.base.py), feedforward agents
remain unaffected.

"""
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import make as gym_make
import psutil


def build_and_train(config):
    cpus = psutil.Process().cpu_affinity()  # should return a list or a tuple
    print(cpus)
    affinity = dict(
        cuda_idx=None,  # whichever one you have
        master_cpus=cpus,
        workers_cpus=list([x] for x in cpus),  # If just one cpu per worker
        set_affinity=True,  # can set to False if you want to turn off rlpyt assigning the psutil cpu_affinity
    )
    sampler = CpuSampler(
        EnvCls=gym_make, **config['sampler']
    )
    algo = SAC(**config['algo'])
    agent = SacAgent(**config['agent'])
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config['runner'],
    )
    with logger_context(**config['logger']):
        runner.train(max_return=400)


if __name__ == "__main__":
    default_configuration = {
        'general':
            {
                # 'sampler_type': 'SerialSampler',  # CpuSampler
                'sampler_type': 'CpuSampler',  # CpuSampler
                'algo': 'SAC'
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
                # 'demonstrations_path': '/home/sagiv/Documents/HUJI/Tsevi/RL/rlpyt/data/lunar_demo.pkl',
                # 'expert_ratio': 0.2,
                # 'expert_discount': 0.8
            },
        'sampler':
            {
                'env_kwargs':
                    {
                        'id': 'LunarLanderContinuous-v2'
                    },
                'eval_env_kwargs':
                    {
                        'id': 'LunarLanderContinuous-v2'
                    },
                'max_decorrelation_steps': 5000,  # Random sampling an action to bootstrap
                'eval_max_steps': 1e5,
                'eval_max_trajectories': 10,

                'batch_T': 512,  # Environment steps per worker in batch
                'batch_B': 56,  # Total environments and agents
                'eval_n_envs': 5,
            },
        'runner':
            {
                'n_steps': 100000,  # Total environment steps
                'log_interval_steps': 5000,
            },
        'logger':
            {
                'log_dir': '/home/sagiv/Documents/HUJI/Tsevi/RL/rlpyt/data/parallel',
                'run_ID': 1212,
                'name': 'SAC',
                'snapshot_mode': 'last',
                'use_summary_writer': True
            }
    }
    build_and_train(
        default_configuration
    )

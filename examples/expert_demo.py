import numpy as np
import torch

from rlpyt.replays.non_sequence.time_limit import TlUniformReplayBuffer
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.envs.gym import make as gym_make
# from rlpyt.agents.classic.pid_agent import PIDAgent
# from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.agents.classic.pi_agent import PIAgent
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.collections import NamedTupleSchema
# from rlpyt.replays.non_sequence.uniform import UniformReplayBuffer
from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
import pickle

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
    SamplesToBuffer._fields + ("timeout",))


def samples_to_buffer(samples):
    """Defines how to add data from sampler into the replay buffer. Called
    in optimize_agent() if samples are provided to that method."""
    samples_to_buffer = SamplesToBuffer(
        observation=samples.env.observation,
        action=samples.agent.action,
        reward=samples.env.reward,
        done=samples.env.done,
    )
    samples_to_buffer = SamplesToBufferTl(*samples_to_buffer,
                                          timeout=samples.env.env_info.timeout)
    return samples_to_buffer

if __name__ == "__main__":
    sampler_kwargs = \
        {
            'env_kwargs':
                {
                    # 'id': 'LunarLanderContinuous-v2'
                    'id': 'gym_flySim:flySim-v0',
                    'random_start': True
                },
            'eval_env_kwargs':
                {
                    'id': 'gym_flySim:flySim-v0',
                    'random_start': True
                    # 'id': 'LunarLanderContinuous-v2'
                },
            'max_decorrelation_steps': 0,  # Random sampling an action to bootstrap
            'eval_max_steps': 880,
            'eval_max_trajectories': 2,

            'batch_T': 128,  # Environment steps per worker in batch
            # 'batch_B': 2,  # Total environments and agents
            'batch_B': 8,  # Total environments and agents
            # 'eval_n_envs': 10,
        }
    # agent = PIDAgent()
    agent = PIAgent()
    # agent = SacAgent()
    # sampler = SerialSampler(
    #     EnvCls=gym_make, **sampler_kwargs
    # )
    sampler = CpuSampler(
        EnvCls=gym_make, **sampler_kwargs
    )
    # examples = sampler.initialize(agent=agent)
    examples = sampler.initialize(
        agent=agent,  # Agent gets initialized in sampler.
        affinity = dict(cuda_idx=None, workers_cpus=[0,1,2,3]),
        # affinity=dict(cuda_idx=None, workers_cpus=[0, 1]),
        seed= 0,
        bootstrap_value=False,
        traj_info_kwargs=dict(discount=1),
        rank=0,
    )
    # samples = samples_to_buffer(samples)
    # samples = SamplesToBuffer(
    #     observation=samples["observation"],
    #     action=samples["action"],
    #     reward=samples["reward"],
    #     done=samples["done"],
    # )
    example_to_buffer = SamplesToBuffer(
        observation=examples["observation"],
        action=examples["action"],
        reward=examples["reward"],
        done=examples["done"],
    )
    example_to_buffer = SamplesToBufferTl(*example_to_buffer,
                                          timeout=examples["env_info"].timeout)
    replay_kwargs = dict(
        example=example_to_buffer,
        size=3072,
        B=sampler_kwargs['batch_B'],
        n_step_return=1,
    )

    expert_buffer = TlUniformReplayBuffer(**replay_kwargs)
    while not expert_buffer._buffer_full:
        samples,traj_infos = sampler.obtain_samples(0)
        samples = samples_to_buffer(samples)
        expert_buffer.append_samples(samples)
        print(f'samples: {len(samples)};  cursor: {expert_buffer.t}/{expert_buffer.T}')
    sampler.shutdown()
    torch.save(expert_buffer, '../data/fly_demo.pkl')
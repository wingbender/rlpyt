"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""
import torch.nn
import numpy as np

from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.ddpg import DDPG
from rlpyt.agents.qpg.ddpg_agent import DdpgAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.buffer import torchify_buffer, numpify_buffer
from rlpyt.agents.base import AgentInputs


def build_and_train(env_id="LunarLanderContinuous-v2", run_ID=0, cuda_idx=None, n_cpu=1, save_path='./model',
                    agent=None):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_cpu)))
    sampler = CpuSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        batch_T=250,  # Environment steps per worker in batch
        batch_B=4,  # Total environments and agents
        max_decorrelation_steps=10000,  # Random sampling an action to bootstrap
        eval_n_envs=4,
        eval_max_steps=int(4000),
        eval_max_trajectories=16,
    )
    algo = DDPG(target_update_tau=0.001, learning_rate=1e-4, q_learning_rate=1e-3)  # Run with defaults.
    if agent is None:
        agent = DdpgAgent(
            model_kwargs={'hidden_sizes': [300, 400], 'nonlinearity': torch.nn.ReLU, 'f_nonlinearity': torch.nn.Tanh})
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=250000,  # Total environment steps 1000 per batch, 1000 per learning iteration (I think)
        log_interval_steps=1000,
        affinity=affinity,
    )
    config = dict(env_id=env_id)
    name = "ddpg_" + env_id
    log_dir = "my_exp_OU_Noise"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode='last'):
        runner.train()
    torch.save(agent.state_dict(), save_path)
    return agent


def show_agent(env_id, episodes, agent=None, agent_path=None):
    env = gym_make(env_id)
    _ = env.reset()
    if agent is None:
        if isinstance(agent_path, str):
            agent = DdpgAgent(
                model_kwargs={'hidden_sizes': [300, 400], 'nonlinearity': torch.nn.LeakyReLU,
                              'f_nonlinearity': torch.nn.Tanh})
            agent.initialize(env.spaces, share_memory=False,
                             global_B=1, env_ranks=None)
            agent.load_state_dict(torch.load(agent_path))
        else:
            raise FileNotFoundError('Please supply agent path')

    else:
        agent = agent
    for i in range(episodes):
        _ = env.reset()
        a = env.action_space.sample()
        o, r, d, env_info = env.step(a)
        r = np.asarray(r, dtype="float32")  # Must match torch float dtype here.
        agent.reset()
        done = False
        tot_reward = 0
        t = 0
        while not done and t < 250:
            agent_inputs = torchify_buffer(AgentInputs(o, a, np.array([r])))
            action, agent_info = agent.step(*agent_inputs)
            action = numpify_buffer(action)
            o, r, done, info = env.step(action)
            env.render()
            tot_reward += r
            t += 1
        print(f'Total reward for episode {i}: {tot_reward}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='LunarLanderContinuous-v2')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=3)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--n_cpu', help='number of cpu to use ', type=int, default=4)
    parser.add_argument('--save_path', help='path to save final model', type=str, default='./model')
    parser.add_argument('--retrain', help='whether or not to retrain the model', type=bool, default=False)
    parser.add_argument('--display', help='whether or not to show the model', type=str, default=True)
    parser.add_argument('--agent_path', help='Path to existing agent to continue training', type=str,
                        default=None)
    args = parser.parse_args()
    if args.retrain:
        agent = None
        if args.agent_path is not None:
            env = gym_make(args.env_id)
            agent = DdpgAgent(
                model_kwargs={'hidden_sizes': [300, 400], 'nonlinearity': torch.nn.LeakyReLU,
                              'f_nonlinearity': torch.nn.Tanh})
            agent.initialize(env.spaces, share_memory=True,
                             global_B=1, env_ranks=None)
            agent.load_state_dict(torch.load(args.agent_path))
        agent = build_and_train(env_id=args.env_id,
                                run_ID=args.run_ID,
                                cuda_idx=args.cuda_idx,
                                n_cpu=args.n_cpu,
                                save_path=args.save_path + str(args.run_ID),
                                agent=agent)
    if args.display:
        if args.retrain:
            show_agent(args.env_id, 10, agent=agent)
        elif args.save_path is not None:
            show_agent(args.env_id, 10, agent_path=args.save_path + str(args.run_ID))
        else:
            raise FileNotFoundError('You didn\'t supply agent path to show')

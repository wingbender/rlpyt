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

from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.buffer import torchify_buffer, numpify_buffer
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.logging import logger


# def build_and_train(env_id="LunarLanderContinuous-v2", run_ID=0, cuda_idx=None, n_cpu=1, save_name='model',
#                     agent=None, agent_path=None, training_mode='start'):
#     affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_cpu)))
def build_and_train(configuration_dict, training_mode):
    # Run with defaults.
    restart = True
    save_name = configuration_dict['general']['save_name']
    run_ID = configuration_dict['general']['run_ID']
    if training_mode is 'continue':
        saved_session_path = os.path.join('./data', save_name, f'run_{run_ID}', 'params.pkl')
        if os.path.isfile(saved_session_path):
            try:
                print('Loading saved session...')
                data = torch.load(saved_session_path)
                itr = data['itr']  # might be useful for logging / debugging
                cum_steps = data['cum_steps']  # might be useful for logging / debugging
                agent_state_dict = data['agent_state_dict']  # 'model' and 'target' keys
                optimizer_state_dict = data['optimizer_state_dict']
                agent = SacAgent(initial_model_state_dict=agent_state_dict, **configuration_dict['agent'])
                algo = SAC(initial_optim_state_dict=optimizer_state_dict, **configuration_dict['algo'])
                decorrelation_steps = 0
                restart = False
                print('Load successful')
            except:
                raise FileNotFoundError(f'Couldn\'t find a pretrained agent in saved_session_path')
    if restart:
        print('Starting new session')
        # agent = SacAgent(model_kwargs={'hidden_sizes': [64, 64, 32, 32]},  # Pi model.
        #                  q_model_kwargs={'hidden_sizes': [64, 64]},
        #                  v_model_kwargs={'hidden_sizes': [32, 32]}, )
        agent = SacAgent(**configuration_dict['agent'])
        algo = SAC(**configuration_dict['algo'])
        decorrelation_steps = configuration_dict['sampler']['max_decorrelation_steps']
        itr = 0
    del configuration_dict['sampler']['max_decorrelation_steps']
    if configuration_dict['general']['sampler_type'] == 'SerialSampler':
        sampler = SerialSampler(
            EnvCls=gym_make, max_decorrelation_steps=decorrelation_steps, **configuration_dict['sampler']
        )
    elif configuration_dict['general']['sampler_type'] == 'CpuSampler':
        sampler = CpuSampler(
            EnvCls=gym_make, max_decorrelation_steps=decorrelation_steps, **configuration_dict['sampler']
        )
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        **configuration_dict['runner'],
    )
    with logger_context(log_dir='./data/sagiv/',
                        run_ID=configuration_dict['general']['run_ID'],
                        name=configuration_dict['general']['save_name'],
                        snapshot_mode='last'):
        exp_dir = logger.get_snapshot_dir()
        conf_filename = os.path.join(exp_dir, 'conf.json')
        with open(conf_filename, 'w') as f:
            f.writelines(json.dumps(configuration_dict))
        runner.train(itr)
    return agent

    # def show_agent(env_id, episodes, agent=None, save_name=None, run_ID=0):


def evaluate_agent(configuration_dict, episodes, display=True):
    env = gym_make(configuration_dict['sampler']['eval_env_kwargs']['id'])
    _ = env.reset()
    save_name = configuration_dict['general']['save_name']
    run_ID = configuration_dict['general']['run_ID']
    saved_session_path = os.path.join('./data', save_name, f'run_{run_ID}', 'params.pkl')
    if os.path.isfile(saved_session_path):
        try:
            data = torch.load(saved_session_path)
            itr = data['itr']  # might be useful for logging / debugging
            cum_steps = data['cum_steps']  # might be useful for logging / debugging
            agent_state_dict = data['agent_state_dict']  # 'model' and 'target' keys
            optimizer_state_dict = data['optimizer_state_dict']
            agent = SacAgent(initial_model_state_dict=agent_state_dict, **configuration_dict['agent'])
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
    # parser.add_argument('--env_id', help='environment ID', default='LunarLanderContinuous-v2')
    # parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=11)
    # parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    # parser.add_argument('--n_cpu', help='number of cpu to use ', type=int, default=4)
    # parser.add_argument('--save_name', help='path to save final model', type=str, default='sac')
    parser.add_argument('--configuration_path', help='Path to load/save configuration file', type=str,
                        default='./default_configuration')
    parser.add_argument('--train_mode', help='whether or not to restart training for existing model',
                        choices=['start', 'continue', 'infer'],
                        type=str, default='continue')
    parser.add_argument('--display', help='whether or not to show the model', type=str, default=True)
    parser.add_argument('-y', dest='ignore_warning', action='store_true')
    parser.set_defaults(ignore_warning=False)
    args = parser.parse_args()

    default_configuration = {
        'general':
            {
                'run_ID': 11,
                'save_name': 'sac',
                'sampler_type': 'SerialSampler'  # CpuSampler

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
                'replay_size': 5e5
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
                'max_decorrelation_steps': 10000,  # Random sampling an action to bootstrap
                'eval_max_steps': 100000,
                'eval_max_trajectories': 100,

                'batch_T': 1000,  # Environment steps per worker in batch
                'batch_B': 10,  # Total environments and agents
                'eval_n_envs': 100,
            },
        'runner':
            {
                'n_steps': 50000,  # Total environment steps
                'log_interval_steps': 1000,
                'affinity':
                    {
                        'cuda_idx': None,
                        'workers_cpus': [1, 2, 3, 4]
                    }
            },
    }
    if args.configuration_path is not None and \
            args.configuration_path.lower().endswith('.json') and \
            os.path.isfile(args.configuration_path):
        with open(args.configuration_path, 'r+') as conf_file:
            config = json.loads(conf_file)
    else:
        config = default_configuration
        with open(args.configuration_path, 'w+') as conf_file:
            conf_file.writelines(json.dumps(config, indent=4))
    if args.train_mode != 'infer':
        build_and_train(config, args.train_mode)
    if args.display:
        evaluate_agent(config, 10, True)
    else:
        evaluate_agent(config, 100, False)

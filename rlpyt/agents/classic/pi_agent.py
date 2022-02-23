import numpy as np
import torch
from collections import namedtuple
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.nn.parallel import DistributedDataParallelCPU as DDPC  # Deprecated

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.models.qpg.mlp import QofMuMlpModel, PiMlpModel
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import numpify_buffer

## Numpy/Scipy imports
import numpy as np
from numpy import sin, cos, tanh, arcsin
from scipy.spatial.transform import Rotation
from numpy.linalg import norm, inv

AgentInfo = namedarraytuple("AgentInfo", ["dist_info"])
Models = namedtuple("Models", ["pi", "q1", "q2", "v"])
DEG2RAD = 180/np.pi


def body_ang_vel_pqr(angles, angles_dot, get_pqr):
    '''
    Converts change in euler angles to body rates (if get_pqr is True) or body rates to euler rates (if get_pqr is False)
    :param angles: euler angles (np.array[psi,theta,phi])
    :param angles_dot: euler rates or body rates (np.array[d(psi)/dt,d(theta)/dt,d(phi)/dt] or np.array([p,q,r])
    :param get_pqr: whether to get body rates from euler rates or the other way around
    :return: euler rates or body rates (np.array[d(psi)/dt,d(theta)/dt,d(phi)/dt] or np.array([p,q,r])
    '''
    psi = angles[0]
    theta = angles[1]
    psi_p_dot = angles_dot[0]
    theta_q_dot = angles_dot[1]
    phi_r_dot = angles_dot[2]

    if get_pqr:
        p = psi_p_dot - phi_r_dot * sin(theta)
        q = theta_q_dot * cos(psi) + phi_r_dot * sin(psi) * cos(theta)
        r = -theta_q_dot * sin(psi) + phi_r_dot * cos(psi) * cos(theta)
        return np.stack([p, q, r])
    else:
        psi_dot = psi_p_dot + theta_q_dot * (sin(psi) * sin(theta)) / cos(theta) + phi_r_dot * (
                cos(psi) * sin(theta)) / cos(theta)
        theta_dot = theta_q_dot * cos(psi) + phi_r_dot * -sin(psi)
        phi_dot = theta_q_dot * sin(psi) / cos(theta) + phi_r_dot * cos(psi) / cos(theta)
        return np.stack([psi_dot, theta_dot, phi_dot])

def body_ang_vel_pqr_vect(angles, angles_dot, get_pqr):
    '''
    Converts change in euler angles to body rates (if get_pqr is True) or body rates to euler rates (if get_pqr is False)
    :param angles: euler angles (np.array[psi,theta,phi])
    :param angles_dot: euler rates or body rates (np.array[d(psi)/dt,d(theta)/dt,d(phi)/dt] or np.array([p,q,r])
    :param get_pqr: whether to get body rates from euler rates or the other way around
    :return: euler rates or body rates (np.array[d(psi)/dt,d(theta)/dt,d(phi)/dt] or np.array([p,q,r])
    '''
    psi = angles[:,0]
    theta = angles[:,1]
    psi_p_dot = angles_dot[:,0]
    theta_q_dot = angles_dot[:,1]
    phi_r_dot = angles_dot[:,2]

    if get_pqr:
        p = psi_p_dot - phi_r_dot * sin(theta)
        q = theta_q_dot * cos(psi) + phi_r_dot * sin(psi) * cos(theta)
        r = -theta_q_dot * sin(psi) + phi_r_dot * cos(psi) * cos(theta)
        return np.stack([p, q, r])
    else:
        psi_dot = psi_p_dot + theta_q_dot * (sin(psi) * sin(theta)) / cos(theta) + phi_r_dot * (
                cos(psi) * sin(theta)) / cos(theta)
        theta_dot = theta_q_dot * cos(psi) + phi_r_dot * -sin(psi)
        phi_dot = theta_q_dot * sin(psi) / cos(theta) + phi_r_dot * cos(psi) / cos(theta)
        return np.stack([psi_dot, theta_dot, phi_dot],axis=1)


class PIAgent(BaseAgent):
    """Agent for expert demonstrations of LunarLanderContinuous based on PID found manually"""

    def __init__(self, *args,**kwargs):
        """Saves input arguments; network defaults stored within."""
        super().__init__(*args,**kwargs)
        self.pitch_kp = 8/1000  # proportional pitch
        self.pitch_ki = 0.5  # integral pitch
        self.pitch_tgt =-45 * DEG2RAD
        # self.kp_ang = -40.33336571  # proportional angle
        # self.kd_ang = 24.34188735  # derivative angle
        save__init__args(locals())
    def initialize(self, env_spaces, share_memory=False, **kwargs):
        """
        Instantiates the neural net model(s) according to the environment
        interfaces.

        Uses shared memory as needed--e.g. in CpuSampler, workers have a copy
        of the agent for action-selection.  The workers automatically hold
        up-to-date parameters in ``model``, because they exist in shared
        memory, constructed here before worker processes fork. Agents with
        additional model components (beyond ``self.model``) for
        action-selection should extend this method to share those, as well.

        Typically called in the sampler during startup.

        Args:
            env_spaces: passed to ``make_env_to_model_kwargs()``, typically namedtuple of 'observation' and 'action'.
            share_memory (bool): whether to use shared memory for model parameters.
        """
        self.env_model_kwargs = self.make_env_to_model_kwargs(env_spaces)
        # self.model = self.ModelCls(**self.env_model_kwargs,
        #     **self.model_kwargs)
        # if share_memory:
        #     self.model.share_memory()
        #     # Store the shared_model (CPU) under a separate name, in case the
        #     # model gets moved to GPU later:
        #     self.shared_model = self.model
        # if self.initial_model_state_dict is not None:
        #     self.model.load_state_dict(self.initial_model_state_dict)
        self.env_spaces = env_spaces
        self.share_memory = share_memory

    def give_min_itr_learn(self, min_itr_learn):
        self.min_itr_learn = min_itr_learn  # From algo.

    def make_env_to_model_kwargs(self, env_spaces):
        assert len(env_spaces.action.shape) == 1
        return dict(
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.shape[0],
        )

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        # Calculate setpoints (target values)
        observation = numpify_buffer(observation)
        reshape = False
        if len(observation.shape)==1:
            reshape = True
            observation=np.expand_dims(observation,0)
        theta = observation[:,7]
        # TODO  vectorize the PI agent starting from next line
        theta_dot = body_ang_vel_pqr_vect(observation[:,6:9], observation[:,3:6], False)[:,1]  # * DEG2RAD
        delta_phi = theta_dot * self.pitch_kp + (theta - (self.pitch_tgt)) * self.pitch_ki

        # Gym wants them as np array (-1,1)
        a = np.array([delta_phi,0*delta_phi]).T
        a = np.clip(a, -1, +1)

        if reshape:
            a = np.squeeze(a)
            # a = np.expand_dims(a,0)
        a = torch.tensor(a)
        dist_info = DistInfoStd(mean=a, log_std=a*0)

        return AgentStep(action=a, agent_info=AgentInfo(dist_info = dist_info))
    def sample_mode(self, itr):
        """Go into sampling mode."""
        self._mode = "sample"

    def eval_mode(self, itr):
        """Go into evaluation mode.  Example use could be to adjust epsilon-greedy."""
        self._mode = "eval"
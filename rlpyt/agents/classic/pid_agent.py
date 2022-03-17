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


AgentInfo = namedarraytuple("AgentInfo", ["dist_info"])
Models = namedtuple("Models", ["pi", "q1", "q2", "v"])



class PIDAgent(BaseAgent):
    """Agent for expert demonstrations of LunarLanderContinuous based on PID found manually"""

    def __init__(self, *args,**kwargs):
        """Saves input arguments; network defaults stored within."""
        super().__init__(*args,**kwargs)
        self.kp_alt = 27.14893426  # proportional altitude
        self.kd_alt = -17.60568096  # derivative altitude
        self.kp_ang = -40.33336571  # proportional angle
        self.kd_ang = 24.34188735  # derivative angle
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
        alt_tgt = np.abs(observation[:,0])
        ang_tgt = (.25 * np.pi) * (observation[:,0] + observation[:,2])

        # Calculate error values
        alt_error = (alt_tgt - observation[:,1])
        ang_error = (ang_tgt - observation[:,4])

        # Use PID to get adjustments
        alt_adj = self.kp_alt * alt_error + self.kd_alt * observation[:,3]
        ang_adj = self.kp_ang * ang_error + self.kd_ang * observation[:,5]

        # Gym wants them as np array (-1,1)
        a = np.array([alt_adj, ang_adj])
        a = np.clip(a, -1, +1)

        # If the legs are on the ground we made it, kill engines
        a = np.logical_not(observation[:,6:8])*a.T
        # model_inputs = buffer_to((observation, prev_action, prev_reward),
        #                          device=self.device)

        if reshape:
            a = np.squeeze(a)
        a = torch.tensor(a)
        dist_info = DistInfoStd(mean=a, log_std=a*0)

        return AgentStep(action=a, agent_info=AgentInfo(dist_info = dist_info))
    def sample_mode(self, itr):
        """Go into sampling mode."""
        self._mode = "sample"

    def eval_mode(self, itr):
        """Go into evaluation mode.  Example use could be to adjust epsilon-greedy."""
        self._mode = "eval"
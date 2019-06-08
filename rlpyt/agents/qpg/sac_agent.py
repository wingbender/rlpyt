
import torch

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.models.qpg.mlp import MlpQModel, MlpVModel, MlpPiModel
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple


AgentInfo = namedarraytuple("AgentInfo", ["dist_info"])


class SacAgent(BaseAgent):

    shared_pi_model = None

    def __init__(
            self,
            QModelCls=MlpQModel,
            VModelCls=MlpVModel,
            PiModelCls=MlpPiModel,
            q_model_kwargs=None,
            v_model_kwargs=None,
            pi_model_kwargs=None,
            initial_q1_model_state_dict=None,
            initial_q2_model_state_dict=None,
            initial_v_model_state_dict=None,
            initial_pi_model_state_dict=None,
            action_squash=None,  # int or float for max magnitude
            ):
        if q_model_kwargs is None:
            q_model_kwargs = dict(hidden_sizes=[256, 256])
        if v_model_kwargs is None:
            v_model_kwargs = dict(hidden_sizes=[256, 256])
        if pi_model_kwargs is None:
            pi_model_kwargs = dict(hidden_sizes=[256, 256])
        save__init__args(locals())

    def initialize(self, env_spec, share_memory=False):
        env_model_kwargs = self.make_env_to_model_kwargs(env_spec)
        self.q1_model = self.QModelCls(**env_model_kwargs, **self.q_model_kwargs)
        self.q2_model = self.QModelCls(**env_model_kwargs, **self.q_model_kwargs)
        self.v_model = self.VModelCls(**env_model_kwargs, **self.v_model_kwargs)
        self.pi_model = self.PiModelCls(**env_model_kwargs, **self.pi_model_kwargs)
        if share_memory:
            self.pi_model.share_memory()  # Only one needed for sampling.
            self.shared_pi_model = self.pi_model
        if self.initial_q1_model_state_dict is not None:
            self.q1_model.load_state_dict(self.initial_q1_model_state_dict)
        if self.initial_q2_model_state_dict is not None:
            self.q2_model.load_state_dict(self.initial_q2_model_state_dict)
        if self.initial_v_model_state_dict is not None:
            self.v_model.load_state_dict(self.initial_v_model_state_dict)
        if self.initial_pi_model_state_dict is not None:
            self.pi_model.load_state_dict(self.initial_pi_model_state_dict)
        self.target_v_model = self.VModelCls(**env_model_kwargs,
            **self.v_model_kwargs)
        self.target_v_model.load_state_dict(self.v_model.state_dict())
        self.distribution = Gaussian(dim=env_spec.action_space.size,
            squash=self.action_squash)
        self.env_spec = env_spec
        self.env_model_kwargs = env_model_kwargs

    def initialize_cuda(self, cuda_idx=None):
        if cuda_idx is None:
            return  # CPU
        if self.shared_pi_model is not None:
            self.pi_model = self.PiModelCls(**self.env_model_kwargs,
                **self.pi_model_kwargs)
            self.pi_model.load_state_dict(self.shared_pi_model.state_dict())
        self.device = torch.device("cuda", index=cuda_idx)
        self.q1_model.to(self.device)
        self.q2_model.to(self.device)
        self.v_model.to(self.device)
        self.pi_model.to(self.device)
        self.target_v_model.to(self.device)
        logger.log(f"Initialized agent models on device: {self.device}.")

    def make_env_to_model_kwargs(self, env_spec):
        return dict(
            observation_size=env_spec.observation_space.size,
            action_size=env_spec.action_space.size,
            obs_n_dim=len(env_spec.observation_space.shape),
        )

    def q(self, observation, prev_action, prev_reward, action):
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            action), device=self.device)
        q1 = self.q1_model(*model_inputs)
        q2 = self.q2_model(*model_inputs)
        return q1.cpu(), q2.cpu()

    def v(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        v = self.v_model(*model_inputs)
        return v.cpu()

    def pi(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mean, log_std = self.pi_model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action = self.distribution.sample(dist_info)
        log_pi = self.distribution.log_likelihood(action, dist_info)
        log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        return action, log_pi, dist_info  # Action stays on device for q models.

    def target_v(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        target_v = self.target_v_model(*model_inputs)
        return target_v.cpu()

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mean, log_std = self.pi_model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def update_target(self, tau=1):
        update_state_dict(self.target_v_model, self.v_model, tau)

    def parameters(self):
        yield from self.q1_model.parameters()
        yield from self.q2_model.parameters()
        yield from self.v_model.parameters()
        yield from self.pi_model.parameters()

    def parameters_by_model(self):
        return (self.q1_model.parameters(), self.q2_model.parameters(),
            self.v_model.parameters(), self.pi_model.parameters())

    def sync_shared_memory(self):
        if self.shared_pi_model is not self.pi_model:
            self.shared_pi_model.load_state_dict(self.pi_model.state_dict())

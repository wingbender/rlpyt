
import numpy as np
import torch
from collections import namedtuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.replays.non_sequence.uniform import (UniformReplayBuffer,
    AsyncUniformReplayBuffer)
from rlpyt.replays.non_sequence.time_limit import (TlUniformReplayBuffer,
    AsyncTlUniformReplayBuffer)
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_to
from rlpyt.distributions.gaussian import Gaussian
from rlpyt.distributions.gaussian import DistInfo as GaussianDistInfo
from rlpyt.utils.tensor import valid_mean
from rlpyt.algos.utils import valid_from_done
from rlpyt.algos.qpg.sac import SAC
import pickle

OptInfo = namedtuple("OptInfo",
    ["q1Loss", "q2Loss", "piLoss",
    "q1GradNorm", "q2GradNorm", "piGradNorm",
    "q1", "q2", "piMu", "piLogStd", "qMeanDiff", "alpha"])
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
    SamplesToBuffer._fields + ("timeout",))
SamplesFromReplay = namedarraytuple("SamplesFromReplay",
                                    ["agent_inputs", "action", "return_", "done", "done_n",
                                     "target_inputs"])
AgentInputs = namedarraytuple("AgentInputs",
    ["observation", "prev_action", "prev_reward"])


class SACfD(SAC):
    """Soft actor critic algorithm, training from a replay buffer
    with expert demonstrations mixed in each learning step"""

    def __init__(self,expert_ratio = 0.25, demonstrations_path = None,*args, **kw):
        self.expert_ratio = expert_ratio
        self.demonstration_path = demonstrations_path
        super().__init__(*args, **kw)

    def initialize(self, *args, **kw):
        super().initialize(*args, **kw)
        self.initialize_expert_buffer(*args, **kw)

    def async_initialize(self, *args, **kw):
        super().async_initialize(*args, **kw)
        self.initialize_expert_buffer(*args, **kw)
        return self.replay_buffer

    def initialize_expert_buffer(self, examples, batch_spec,*args,**kw):
        """
        Allocates replay buffer using examples and with the fields in `SamplesToBuffer`
        namedarraytuple.
        """
        #
        # example_to_buffer = SamplesToBuffer(
        #     observation=examples["observation"],
        #     action=examples["action"],
        #     reward=examples["reward"],
        #     done=examples["done"],
        # )
        # if not self.bootstrap_timelimit:
        #     ReplayCls = UniformReplayBuffer
        # else:
        #     example_to_buffer = SamplesToBufferTl(*example_to_buffer,
        #         timeout=examples["env_info"].timeout)
        #     ReplayCls = TlUniformReplayBuffer
        # replay_kwargs = dict(
        #     example=example_to_buffer,
        #     size=self.replay_size,
        #     B=batch_spec.B,
        #     n_step_return=self.n_step_return,
        # )
        # if self.ReplayBufferCls is not None:
        #     ReplayCls = self.ReplayBufferCls
        #     logger.log(f"WARNING: ignoring internal selection logic and using"
        #         f" input replay buffer class: {ReplayCls} -- compatibility not"
        #         " guaranteed.")
        # self.expert_buffer = ReplayCls(**replay_kwargs)
        temp_RB  = torch.load(self.demonstration_path) # TODO: loading here only succeeds if I first load on the console, something to do with the loading of SamplesToBufferTl or related to it
        self.expert_buffer = temp_RB
        # with open(self.demonstration_path, 'rb') as f:
        #     self.expert_buffer = pickle.load(f)


    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Extracts the needed fields from input samples and stores them in the 
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).
        """
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info
        for _ in range(self.updates_per_optimize):
            experience_batch_size = int(self.batch_size*(1-self.expert_ratio))
            expert_batch_size = self.batch_size-experience_batch_size
            samples_from_replay = self.replay_buffer.sample_batch(experience_batch_size)
            samples_from_expert = self.expert_buffer.sample_batch(expert_batch_size)
            # samples = SamplesFromReplay(
            #     agent_inputs=AgentInputs(
            #         observation=self.extract_observation(T_idxs, B_idxs),
            #         prev_action=s.action[T_idxs - 1, B_idxs],
            #         prev_reward=s.reward[T_idxs - 1, B_idxs],
            #     ),
            #     action=s.action[T_idxs, B_idxs],
            #     return_=self.samples_return_[T_idxs, B_idxs],
            #     done=self.samples.done[T_idxs, B_idxs],
            #     done_n=self.samples_done_n[T_idxs, B_idxs],
            #     target_inputs=AgentInputs(
            #         observation=self.extract_observation(target_T_idxs, B_idxs),
            #         prev_action=s.action[target_T_idxs - 1, B_idxs],
            #         prev_reward=s.reward[target_T_idxs - 1, B_idxs],
            #     ),
            # )
            samples = torch.cat(samples_from_replay,samples_from_expert) # TODO: This is not working, I have to find a way to combine two namedarraytuple
            losses, values = self.loss(samples)
            q1_loss, q2_loss, pi_loss, alpha_loss = losses

            if alpha_loss is not None:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self._alpha = torch.exp(self._log_alpha.detach())

            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.pi_parameters(),
                self.clip_grad_norm)
            self.pi_optimizer.step()

            # Step Q's last because pi_loss.backward() uses them?
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            q1_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q1_parameters(),
                self.clip_grad_norm)
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            q2_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q2_parameters(),
                self.clip_grad_norm)
            self.q2_optimizer.step()

            grad_norms = (q1_grad_norm, q2_grad_norm, pi_grad_norm)

            self.append_opt_info_(opt_info, losses, grad_norms, values)
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)

        return opt_info

    def samples_to_buffer(self, samples):
        """Defines how to add data from sampler into the replay buffer. Called
        in optimize_agent() if samples are provided to that method."""
        samples_to_buffer = SamplesToBuffer(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
        )
        if self.bootstrap_timelimit:
            samples_to_buffer = SamplesToBufferTl(*samples_to_buffer,
                timeout=samples.env.env_info.timeout)
        return samples_to_buffer

    def loss(self, samples):
        """
        Computes losses for twin Q-values against the min of twin target Q-values
        and an entropy term.  Computes reparameterized policy loss, and loss for
        tuning entropy weighting, alpha.  
        
        Input samples have leading batch dimension [B,..] (but not time).
        """
        agent_inputs, target_inputs, action = buffer_to(
            (samples.agent_inputs, samples.target_inputs, samples.action))

        if self.mid_batch_reset and not self.agent.recurrent:
            valid = torch.ones_like(samples.done, dtype=torch.float)  # or None
        else:
            valid = valid_from_done(samples.done)
        if self.bootstrap_timelimit:
            # To avoid non-use of bootstrap when environment is 'done' due to
            # time-limit, turn off training on these samples.
            valid *= (1 - samples.timeout_n.float())

        q1, q2 = self.agent.q(*agent_inputs, action)
        with torch.no_grad():
            target_action, target_log_pi, _ = self.agent.pi(*target_inputs)
            target_q1, target_q2 = self.agent.target_q(*target_inputs, target_action)
        min_target_q = torch.min(target_q1, target_q2)
        target_value = min_target_q - self._alpha * target_log_pi
        disc = self.discount ** self.n_step_return
        y = (self.reward_scale * samples.return_ +
            (1 - samples.done_n.float()) * disc * target_value)

        q1_loss = 0.5 * valid_mean((y - q1) ** 2, valid)
        q2_loss = 0.5 * valid_mean((y - q2) ** 2, valid)

        new_action, log_pi, (pi_mean, pi_log_std) = self.agent.pi(*agent_inputs)
        if not self.reparameterize:
            new_action = new_action.detach()  # No grad.
        log_target1, log_target2 = self.agent.q(*agent_inputs, new_action)
        min_log_target = torch.min(log_target1, log_target2)
        prior_log_pi = self.get_action_prior(new_action.cpu())

        if self.reparameterize:
            pi_losses = self._alpha * log_pi - min_log_target - prior_log_pi
        else:
            raise NotImplementedError

        # if self.policy_output_regularization > 0:
        #     pi_losses += self.policy_output_regularization * torch.mean(
        #         0.5 * pi_mean ** 2 + 0.5 * pi_log_std ** 2, dim=-1)
        pi_loss = valid_mean(pi_losses, valid)

        if self.target_entropy is not None and self.fixed_alpha is None:
            alpha_losses = - self._log_alpha * (log_pi.detach() + self.target_entropy)
            alpha_loss = valid_mean(alpha_losses, valid)
        else:
            alpha_loss = None

        losses = (q1_loss, q2_loss, pi_loss, alpha_loss)
        values = tuple(val.detach() for val in (q1, q2, pi_mean, pi_log_std))
        return losses, values

    def get_action_prior(self, action):
        if self.action_prior == "uniform":
            prior_log_pi = 0.0
        elif self.action_prior == "gaussian":
            prior_log_pi = self.action_prior_distribution.log_likelihood(
                action, GaussianDistInfo(mean=torch.zeros_like(action)))
        return prior_log_pi

    def append_opt_info_(self, opt_info, losses, grad_norms, values):
        """In-place."""
        q1_loss, q2_loss, pi_loss, alpha_loss = losses
        q1_grad_norm, q2_grad_norm, pi_grad_norm = grad_norms
        q1, q2, pi_mean, pi_log_std = values
        opt_info.q1Loss.append(q1_loss.item())
        opt_info.q2Loss.append(q2_loss.item())
        opt_info.piLoss.append(pi_loss.item())
        opt_info.q1GradNorm.append(torch.tensor(q1_grad_norm).item())  # backwards compatible
        opt_info.q2GradNorm.append(torch.tensor(q2_grad_norm).item())  # backwards compatible
        opt_info.piGradNorm.append(torch.tensor(pi_grad_norm).item())  # backwards compatible
        opt_info.q1.extend(q1[::10].numpy())  # Downsample for stats.
        opt_info.q2.extend(q2[::10].numpy())
        opt_info.piMu.extend(pi_mean[::10].numpy())
        opt_info.piLogStd.extend(pi_log_std[::10].numpy())
        opt_info.qMeanDiff.append(torch.mean(abs(q1 - q2)).item())
        opt_info.alpha.append(self._alpha.item())

    def optim_state_dict(self):
        return dict(
            pi_optimizer=self.pi_optimizer.state_dict(),
            q1_optimizer=self.q1_optimizer.state_dict(),
            q2_optimizer=self.q2_optimizer.state_dict(),
            alpha_optimizer=self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            log_alpha=self._log_alpha.detach().item(),
            replay_buffer=self.replay_buffer,
            expert_buffer = self.expert_buffer
        )
    # TODO: add to this dictionary parameters critical to resume training

    def load_optim_state_dict(self, state_dict):
        self.pi_optimizer.load_state_dict(state_dict["pi_optimizer"])
        self.q1_optimizer.load_state_dict(state_dict["q1_optimizer"])
        self.q2_optimizer.load_state_dict(state_dict["q2_optimizer"])
        if self.alpha_optimizer is not None and state_dict["alpha_optimizer"] is not None:
            self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        with torch.no_grad():
            self._log_alpha[:] = state_dict["log_alpha"]
            self._alpha = torch.exp(self._log_alpha.detach())
        self.replay_buffer = state_dict["replay_buffer"]
        self.expert_buffer = state_dict["expert_buffer"]
    # TODO: load here parameters critical to resume training
#ref: https://github.com/wjh720/QPLEX/blob/master/pymarl-master/src/learners/dmaq_qatten_learner.py

import copy

import numpy as np
import torch as th
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
from torch.optim import RMSprop
from torch.optim import AdamW

class Qplex:

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        state_dim = args.state_shape

        if args.mixer is not None:
            if args.mixer == "dmaq":
                self.mixer = DMAQer(args, state_dim=state_dim)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)


        if self.args.optim=="rmsprop":
            self.optimiser = RMSprop(
            params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif self.args.optim=="adamw":
            self.optimiser = AdamW(
                params=self.params, lr=args.lr)
        else:
            raise ValueError("optimiser {} not recognised.".format(args.optim))


        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.n_actions = self.args.n_actions

        self.trainable_params_print()

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params, show_demo=False, save_data=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        last_actions_onehot = th.cat([th.zeros_like(
            actions_onehot[:, 0].unsqueeze(1)), actions_onehot], dim=1)  # last_actions

        mac.init_hidden(batch.batch_size)
        initial_hidden = mac.hidden_states.clone().detach()


        initial_hidden = initial_hidden.reshape(
            -1, initial_hidden.shape[-1]).to(self.args.device)
        input_here = th.cat((batch["obs"], last_actions_onehot),
                            dim=-1)
        if self.args.obs_agent_id:
            id=th.eye(self.args.n_agents, device=self.args.device).expand(batch.batch_size, batch.max_seq_length, -1, -1)
            input_here = th.cat((input_here, id), dim=-1)
        input_here = input_here.permute(0, 2, 1, 3).to(self.args.device)

        mac_out, _, _ = mac.agent.forward(
            input_here.clone().detach(), initial_hidden.clone().detach())


        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(
            mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals -
                      chosen_action_qvals).detach().cpu().numpy()

        # Calculate the Q-Values necessary for the target
        self.target_mac.init_hidden(batch.batch_size)
        initial_hidden_target = self.target_mac.hidden_states.clone().detach()
        initial_hidden_target = initial_hidden_target.reshape(
            -1, initial_hidden_target.shape[-1]).to(self.args.device)
        target_mac_out, target_mac_hidden_states, _ = self.target_mac.agent.forward(
            input_here.clone().detach(), initial_hidden_target.clone().detach())

        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_chosen_qvals = th.gather(
                target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(
                3).shape + (self.n_actions,)).to(self.args.device)
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(
                3, cur_max_actions, 1)
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(
                target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if mixer is not None:
            # dmaq_general
            ans_chosen = mixer(chosen_action_qvals,
                               batch["state"][:, :-1], is_v=True)
            ans_adv = mixer(chosen_action_qvals, batch["state"][:, :-1],
                            actions=actions_onehot, max_q_i=max_action_qvals, is_v=False)
            chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                # dmaq_general
                target_chosen = self.target_mixer(
                    target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                target_adv = self.target_mixer(
                    target_chosen_qvals, batch["state"][:, 1:], actions=cur_max_actions_onehot, max_q_i=target_max_qvals, is_v=False)
                target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(
                    target_max_qvals, batch["state"][:, 1:], is_v=True)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * \
            (1 - terminated) * target_max_qvals

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]), np.squeeze(q_i_data[:, 0]),
                  np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (
                save_data[0], save_data[1]), np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        q_loss = (masked_td_error**2).sum() / mask.sum()
        loss = q_loss

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            params, self.args.grad_norm_clip)
        optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:

            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals *
                                                  mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat(
                "target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("reward_mean", (rewards * mask).sum().item() / mask.sum(), t_env)
            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, test_data, show_demo=False, save_data=None):
        self.sub_train(
            batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params, show_demo=show_demo, save_data=save_data)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.to(self.args.device)
            self.target_mixer.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))



    def trainable_params_print(self):
        self.logger.console_logger.info(f"Agent NN params:{sum(param.numel() for param in self.mac.parameters())}")
        self.logger.console_logger.info(f"Mixer NN params:{sum(param.numel() for param in self.mixer.parameters())}")
        self.logger.console_logger.info(f"In total, Optimizer model_opt has {sum(param.numel() for param in self.params)} variables.")


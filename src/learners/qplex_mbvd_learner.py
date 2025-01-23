import copy

import numpy as np
import torch as th
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
from torch.optim import RMSprop
from torch.optim import AdamW

from modules.mbvd_model.model import state_space_model


class Qplex_MBVDLearner:

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None

        if args.mixer is not None:
            if args.mixer == "dmaq":
                self.mixer = DMAQer(args, state_dim=args.state_shape * 2)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.model = state_space_model(args.rnn_hidden_dim, args)
        self.params += list(self.model.parameters())

        if self.args.optim == "rmsprop":
            self.optimiser = RMSprop(
                params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif self.args.optim == "adamw":
            self.optimiser = AdamW(
                params=self.params, lr=args.lr)
        else:
            raise ValueError("optimiser {} not recognised.".format(args.optim))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_actions = self.args.n_actions

        self.trainable_params_print()

    def model_train(self, batch, input):
        vae_mask = batch["filled"].float()
        avail_actions = batch["avail_actions"]
        prior_vae_mask = th.cat([th.zeros_like(vae_mask[:, 0]).unsqueeze(1), vae_mask[:, :-1]], dim=1)
        recon_obs, posterior_mu, posterior_logvar, sample_state = self.model.posterior_forward(input)

        # L_FA avail_action_decoder + posterior_encoder
        recon_avail_actions = self.model.get_avail_action(
            sample_state.view(*sample_state.size()[:2], self.args.n_agents, -1))
        recon_avail_actions_dist = th.distributions.Bernoulli(logits=recon_avail_actions)
        avail_actions_loss = -recon_avail_actions_dist.log_prob(avail_actions.float())
        avail_actions_loss = (avail_actions_loss * vae_mask.unsqueeze(-1).expand_as(
            avail_actions_loss)).sum() / vae_mask.unsqueeze(-1).expand_as(avail_actions_loss).sum()
        # L_FA

        # L_RC posterior_encoder + posterior_decoder
        obs_mse_loss = (recon_obs - input).pow(2)
        obs_mse_loss = (obs_mse_loss * vae_mask.unsqueeze(-1).expand_as(obs_mse_loss)).sum() / vae_mask.unsqueeze(
            -1).expand_as(obs_mse_loss).sum()
        # L_RC

        past_onehot_action = th.cat(
            [th.zeros_like(batch["actions_onehot"][:, 0]).unsqueeze(1), batch["actions_onehot"][:, :-1]], dim=1)
        past_sample_state = th.cat(
            [th.zeros([sample_state.size(0), 1, sample_state.size(2)]).to(sample_state.device), sample_state[:, :-1]],
            dim=-2)
        length = vae_mask.size(1) - self.args.k_step + 1
        prior_sample_state = past_sample_state[:, : length]
        total_prior_state_mse_loss = th.tensor(0.).cuda()
        total_prior_action_loss = th.tensor(0.).cuda()
        total_kld_loss = th.tensor(0.).cuda()
        total_prior_kld_loss = th.tensor(0.).cuda()
        for k in range(self.args.k_step):  # k_step=1
            recon_prior_state, recon_action, prior_mu, prior_logvar, next_prior_sample_state = self.model.prior_forward(
                prior_sample_state, past_onehot_action[:, k: length + k])

            prior_kld_loss = 1 + prior_logvar - prior_mu.pow(2) - prior_logvar.exp()
            # KL(q|p)
            if self.args.kl_balance:  # default: True
                kld_lhs_loss = posterior_logvar[:, k: length + k].detach() - prior_logvar + 1 - (
                        posterior_logvar[:, k: length + k].exp().detach() + (
                        posterior_mu[:, k: length + k].detach() - prior_mu).pow(2)) / (
                                       prior_logvar.exp() + 1e-8)
                kld_rhs_loss = posterior_logvar[:, k: length + k] - prior_logvar.detach() + 1 - (
                        posterior_logvar[:, k: length + k].exp() + (
                        posterior_mu[:, k: length + k] - prior_mu.detach()).pow(2)) / (
                                       prior_logvar.exp().detach() + 1e-8)
                kld_loss = self.args.prior_alpha * kld_lhs_loss + (1 - self.args.prior_alpha) * kld_rhs_loss
            else:
                kld_loss = posterior_logvar[:, k: length + k] - prior_logvar + 1 - (
                        posterior_logvar[:, k: length + k].exp() + (posterior_mu[:, k: length + k] - prior_mu).pow(
                    2)) / prior_logvar.exp()
            if self.args.kl_regular:  # default: False
                kld_loss += 1 + posterior_logvar[:, k: length + k] - posterior_mu[:, k: length + k].pow(
                    2) - posterior_logvar[:, k: length + k].exp()
            # L_KL

            # L^{Prior}_{RC} prior all + posterior encoder!!!
            prior_state_mse_loss = (recon_prior_state - prior_sample_state.detach()).pow(2)
            prior_action_loss = - past_onehot_action[:, k: length + k] * recon_action
            # L^{Prior}_{RC}

            total_kld_loss += (- 0.5 * kld_loss * vae_mask[:, k: length + k].expand_as(kld_loss)).sum() / vae_mask[:,
                                                                                                          k: length + k].expand_as(
                kld_loss).sum()
            total_prior_kld_loss += (- 0.5 * prior_kld_loss * prior_vae_mask[:, k: length + k].expand_as(
                prior_kld_loss)).sum() / prior_vae_mask[:, k: length + k].expand_as(prior_kld_loss).sum()
            total_prior_state_mse_loss += (prior_state_mse_loss * prior_vae_mask[:, k: length + k].expand_as(
                prior_state_mse_loss)).sum() / prior_vae_mask[:, k: length + k].expand_as(prior_state_mse_loss).sum()
            total_prior_action_loss += (prior_action_loss * prior_vae_mask[:, k: length + k].unsqueeze(-1).expand_as(
                prior_action_loss)).sum() / prior_vae_mask[:, k: length + k].unsqueeze(-1).expand_as(
                prior_action_loss).sum()

            prior_sample_state = next_prior_sample_state

        total_kld_loss = total_kld_loss * self.args.mbvd_beta
        total_prior_kld_loss = total_prior_kld_loss * self.args.mbvd_beta2

        total_prior_state_mse_loss = total_prior_state_mse_loss / self.args.k_step
        total_prior_action_loss = total_prior_action_loss / self.args.k_step
        total_kld_loss = total_kld_loss / self.args.k_step
        total_prior_kld_loss = total_prior_kld_loss / self.args.k_step
        return posterior_mu, posterior_logvar, sample_state, obs_mse_loss, total_kld_loss, total_prior_kld_loss, total_prior_state_mse_loss, total_prior_action_loss, avail_actions_loss, next_prior_sample_state

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,
                  show_demo=False, save_data=None):
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
            id = th.eye(self.args.n_agents, device=self.args.device).expand(batch.batch_size, batch.max_seq_length, -1,
                                                                            -1)
            input_here = th.cat((input_here, id), dim=-1)
        input_here = input_here.permute(0, 2, 1, 3).to(self.args.device)

        mac_out, mac_hidden_states, _ = mac.agent.forward(
            input_here.clone().detach(), initial_hidden.clone().detach())
        mac_hidden_states = mac_hidden_states.reshape(batch.batch_size, batch.max_seq_length, self.args.n_agents, -1)

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

        target_mac_hidden_states = target_mac_hidden_states.reshape(batch.batch_size, batch.max_seq_length,
                                                                    self.args.n_agents, -1)


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

        posterior_mu, posterior_logvar, sample_state, obs_mse_loss, kld_loss, prior_kld_loss, prior_state_mse_loss, prior_action_loss, avail_actions_loss, next_prior_sample_state = self.model_train(
            batch, target_mac_hidden_states)

        state_loss = obs_mse_loss + kld_loss
        prior_loss = prior_kld_loss + prior_state_mse_loss + prior_action_loss
        current_rollout_depth = self.args.rollout_depth

        i_latent_states = posterior_mu.clone()
        i_avail_actions = batch["avail_actions"].clone()
        i_inputs = mac_hidden_states.clone()
        bs = i_latent_states.size(0)
        sl = i_latent_states.size(1)
        rollout_states = [posterior_mu]

        for h in range(current_rollout_depth):
            with th.no_grad():

                # dummy data preprocess
                zero_mask = th.all(i_avail_actions == 0, dim=-1)
                add_ten = th.zeros_like(i_avail_actions, device=self.args.device)
                add_ten[..., 0] = 1  # only no-op avaliable in smac
                i_avail_actions = i_avail_actions + (zero_mask.unsqueeze(-1) * add_ten)

                i_qs, i_chosen_actions = self.mac.rollout_act(i_avail_actions, i_inputs, t_env=t_env,
                                                              test_mode=self.args.img_greedy,
                                                              rollout=self.args.img_greedy)
                i_onehot_chosen_actions = th.zeros(bs, sl, self.args.n_agents, self.args.n_actions).cuda().scatter_(3,
                                                                                                                    i_chosen_actions.unsqueeze(
                                                                                                                        -1),
                                                                                                                    1).float()
            i_prior_mu, i_prior_logvar = self.model.prior_encode(i_latent_states, i_onehot_chosen_actions)

            if self.args.rollout_random_scale == 0:
                i_next_latent_states = i_prior_mu
            else:
                eps = th.randn_like(i_prior_logvar) * self.args.rollout_random_scale
                i_next_latent_states = i_prior_mu + eps * th.exp(0.5 * i_prior_logvar).detach()

            i_next_inputs = self.model.posterior_decode(i_next_latent_states)
            if self.args.auxiliary_task:
                with th.no_grad():
                    i_avail_actions = self.model.get_avail_action(
                        i_next_latent_states.view(*i_next_latent_states.size()[:2], self.args.n_agents, -1))
                    i_avail_actions = th.distributions.Bernoulli(logits=i_avail_actions).probs
                    i_avail_actions[i_avail_actions >= 0.5] = 1
                    i_avail_actions[i_avail_actions < 0.5] = 0
                    i_avail_actions = i_avail_actions.long()

            i_inputs = i_next_inputs
            i_latent_states = i_next_latent_states
            rollout_states.append(i_prior_mu)

        rollout_states = th.stack(rollout_states, dim=2)
        traj_encode = self.model.get_traj_encode(rollout_states)
        rollout_latent_states = th.cat([batch['state'], traj_encode], dim=-1)


        # Mix
        if mixer is not None:
            # dmaq_general
            ans_chosen = mixer(chosen_action_qvals,
                               rollout_latent_states[:, :-1], is_v=True)
            ans_adv = mixer(chosen_action_qvals, rollout_latent_states[:, :-1],
                            actions=actions_onehot, max_q_i=max_action_qvals, is_v=False)
            chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                # dmaq_general
                target_chosen = self.target_mixer(
                    target_chosen_qvals, rollout_latent_states[:, 1:], is_v=True)
                target_adv = self.target_mixer(
                    target_chosen_qvals, rollout_latent_states[:, 1:], actions=cur_max_actions_onehot,
                    max_q_i=target_max_qvals, is_v=False)
                target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(
                    target_max_qvals, rollout_latent_states[:, 1:], is_v=True)

        # Calculate 1-step Q-Learning targets

        targets = rewards + self.args.gamma * \
                  (1 - terminated) * target_max_qvals

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]),
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
        q_loss = (masked_td_error ** 2).sum() / mask.sum()
        loss = prior_loss + state_loss + q_loss
        if self.args.auxiliary_task:
            loss += avail_actions_loss

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
            self.logger.log_stat("mbvd/L_RC", obs_mse_loss.item(), t_env)
            self.logger.log_stat("mbvd/KL(q|p)", kld_loss.item(), t_env)
            self.logger.log_stat("mbvd/KL(p|N)", prior_kld_loss.item(), t_env)
            self.logger.log_stat("mbvd/L_RC_prior", (prior_state_mse_loss + prior_action_loss).item(), t_env)
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
            batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params, show_demo=show_demo,
            save_data=save_data)
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
        self.model.cuda()
        if self.mixer is not None:
            self.mixer.to(self.args.device)
            self.target_mixer.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.model.state_dict(), "{}/vae.th".format(path))


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
        self.model.load_state_dict(th.load("{}/vae.th".format(path), map_location=lambda storage, loc: storage))

    def trainable_params_print(self):
        self.logger.console_logger.info(f"Agent NN params:{sum(param.numel() for param in self.mac.parameters())}")
        self.logger.console_logger.info(f"Mixer NN params:{sum(param.numel() for param in self.mixer.parameters())}")
        self.logger.console_logger.info(f"VAE params:{sum(param.numel() for param in self.model.parameters())}")
        self.logger.console_logger.info(
            f"In total, Optimizer model_opt has {sum(param.numel() for param in self.params)} variables.")

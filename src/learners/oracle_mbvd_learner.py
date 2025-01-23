import copy
import os

import torch as th
import numpy as np
from torch.optim import RMSprop

from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.mbvd_model.oracle_model import oracle_ssm
from utils.tools import symlog, symexp
from components.dis_cum_rew import discounted_cumulative_rewards
from components.info_combiner import Dynamic_balance_Gate_v3


class OracleMBVDLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        hyper_input_dim = int(np.prod(args.state_shape)) if args.use_combiner else int(np.prod(args.state_shape)) * 2

        if args.mixer is not None:
            self.mixer = QMixer(args, hyper_input_dim)
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
        else:
            raise ValueError("Mixer not exist.")

        self.model = oracle_ssm(args.rnn_hidden_dim, args)
        self.params += list(self.model.parameters())

        if args.use_combiner:
            self.combiner = Dynamic_balance_Gate_v3(args)
            self.params += list(self.combiner.parameters())

        self.rl_optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.mse = th.nn.MSELoss(reduction="none")
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.log_z_t = -self.args.z_log_interval - 1

        if self.args.z_log_interval != 0:
            path = os.path.join(self.args.local_results_path, "datas", self.args.unique_token)
            print(f"save latent z to {path}")
            self.path_post = path + "/post"
            if not os.path.exists(self.path_post):
                os.makedirs(self.path_post)
            self.path_prior = path + "/prior"
            if not os.path.exists(self.path_prior):
                os.makedirs(self.path_prior)

        self.validation_batch = None
        self.validation_tau = None

        if args.env=="stag_hunt":
            self.noop_id = 4
        elif args.env=="sc2":
            self.noop_id = 0
        self.trainable_params_print()


    def model_train(self, batch, input):
        vae_mask = batch["filled"].float()
        avail_actions = batch["avail_actions"]
        reward = batch["reward"]
        prior_vae_mask = th.cat([th.zeros_like(vae_mask[:, 0]).unsqueeze(1), vae_mask[:, :-1]], dim=1)
        recon_obs, posterior_mu, posterior_logvar, sample_state = self.model.posterior_forward(input)


        # L_FA avail_action_decoder + posterior_encoder
        recon_avail_actions = self.model.get_avail_action(sample_state.view(*sample_state.size()[:2], self.args.n_agents, -1))
        recon_avail_actions_dist = th.distributions.Bernoulli(logits=recon_avail_actions)
        avail_actions_loss = -recon_avail_actions_dist.log_prob(avail_actions.float())
        avail_actions_loss = (avail_actions_loss * vae_mask.unsqueeze(-1).expand_as(avail_actions_loss)).sum() / vae_mask.unsqueeze(-1).expand_as(avail_actions_loss).sum()
        # L_FA

        # L_RC posterior_encoder + posterior_decoder
        obs_mse_loss = (recon_obs - input).pow(2)
        obs_mse_loss = (obs_mse_loss * vae_mask.unsqueeze(-1).expand_as(obs_mse_loss)).sum() / vae_mask.unsqueeze(-1).expand_as(obs_mse_loss).sum()
        # L_RC


        past_onehot_action = th.cat([th.zeros_like(batch["actions_onehot"][:, 0]).unsqueeze(1), batch["actions_onehot"][:, :-1]], dim=1)
        past_sample_state = th.cat([th.zeros([sample_state.size(0), 1, sample_state.size(2)]).to(sample_state.device), sample_state[:, :-1]], dim=-2)
        length = vae_mask.size(1) - self.args.k_step + 1
        prior_sample_state = past_sample_state[:, : length]
        total_kld_loss = th.tensor(0.).cuda()
        total_prior_kld_loss = th.tensor(0.).cuda()
        for k in range(self.args.k_step): # k_step=1
            prior_mu, prior_logvar, next_prior_sample_state = self.model.prior_forward(prior_sample_state, past_onehot_action[:, k: length + k])

            # L_KL prior + posterior
                # KL(p|N(0,1))
            prior_kld_loss = 1 + prior_logvar - prior_mu.pow(2) - prior_logvar.exp()
                # KL(q|p)
            if self.args.kl_balance: # default: True
                kld_lhs_loss = posterior_logvar[:, k: length + k].detach() - prior_logvar + 1 - (posterior_logvar[:, k: length + k].exp().detach() + (posterior_mu[:, k: length + k].detach() - prior_mu).pow(2)) / (prior_logvar.exp()+1e-8)
                kld_rhs_loss = posterior_logvar[:, k: length + k] - prior_logvar.detach() + 1 - (posterior_logvar[:, k: length + k].exp() + (posterior_mu[:, k: length + k] - prior_mu.detach()).pow(2)) / (prior_logvar.exp().detach()+1e-8)
                kld_loss = self.args.prior_alpha * kld_lhs_loss + (1 - self.args.prior_alpha) * kld_rhs_loss
            else:
                kld_loss = posterior_logvar[:, k: length + k] - prior_logvar + 1 - (posterior_logvar[:, k: length + k].exp() + (posterior_mu[:, k: length + k] - prior_mu).pow(2)) / prior_logvar.exp()

            if self.args.kl_regular: # default: False
                kld_loss += 1 + posterior_logvar[:, k: length + k] - posterior_mu[:, k: length + k].pow(2) - posterior_logvar[:, k: length + k].exp()
            # L_KL

            total_kld_loss += (- 0.5 * kld_loss * vae_mask[:, k: length + k].expand_as(kld_loss)).sum() / vae_mask[:, k: length + k].expand_as(kld_loss).sum()
            total_prior_kld_loss += (- 0.5 * prior_kld_loss * prior_vae_mask[:, k: length + k].expand_as(prior_kld_loss)).sum() / prior_vae_mask[:, k: length + k].expand_as(prior_kld_loss).sum()
            recon_rew = self.model.get_rew(next_prior_sample_state)
            rew = th.cat([th.zeros_like(reward[:, 0]).unsqueeze(1), reward[:, :-1]], dim=1)
            cum_rew = discounted_cumulative_rewards(symlog(rew), self.args.gamma)
            rew_loss = (cum_rew - recon_rew).pow(2)
            prior_sample_state = next_prior_sample_state

        total_rew_loss = rew_loss[prior_vae_mask == 1].mean() * self.args.rew_beta
        beta_total_kld_loss = total_kld_loss * self.args.mbvd_beta
        total_prior_kld_loss = total_prior_kld_loss * self.args.mbvd_beta2

        return posterior_mu, posterior_logvar, sample_state, obs_mse_loss, beta_total_kld_loss, avail_actions_loss, next_prior_sample_state, total_rew_loss, total_prior_kld_loss



    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, test_data=None):

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        entropy = batch["entropy"]
        alive_mask = batch["alive_agent_mask"]


        # Calculate estimated Q-Values
        mac_out = []
        mac_hidden_states = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            mac_hidden_states.append(self.mac.hidden_states)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_hidden_states = th.stack(mac_hidden_states, dim=1)
        mac_hidden_states = mac_hidden_states.reshape(batch.batch_size, self.args.n_agents, batch.max_seq_length, -1).transpose(1, 2) #btav

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_mac_hidden_states = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            target_mac_hidden_states.append(self.target_mac.hidden_states)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_mac_hidden_states = th.stack(target_mac_hidden_states, dim=1)
        target_mac_hidden_states = target_mac_hidden_states.reshape(batch.batch_size, self.args.n_agents, batch.max_seq_length, -1).transpose(1, 2) #btav


        if self.args.validation:
            if self.validation_batch == None:
                self.validation_batch=copy.deepcopy(batch)
            if self.validation_tau == None:
                self.validation_tau=target_mac_hidden_states.detach().clone()

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        posterior_mu, posterior_logvar, sample_state, obs_mse_loss, kld_loss, avail_actions_loss, next_prior_sample_state, rew_loss, prior_kld = self.model_train(batch, target_mac_hidden_states)

        state_loss = obs_mse_loss + kld_loss + rew_loss + prior_kld
        current_rollout_depth = self.args.rollout_depth

        i_latent_states = posterior_mu.clone()
        i_avail_actions = batch["avail_actions"].clone()
        i_inputs = mac_hidden_states.clone()
        bs = i_latent_states.size(0)
        sl = i_latent_states.size(1)
        rollout_states = [posterior_mu]

        img_chosen_acts = []
        #img_rew = []
        dead_agent_act = th.zeros(1, device=self.args.device)
        all_zero_ava_prev = th.zeros(1, device=self.args.device)
        all_zero_ava_after = th.zeros(1, device=self.args.device)

        for h in range(current_rollout_depth):
            with th.no_grad():

                # dummy data preprocess
                zero_mask = th.all(i_avail_actions==0, dim=-1)
                all_zero_ava_prev += zero_mask.sum()

                add_ten = th.zeros_like(i_avail_actions, device=self.args.device, dtype=th.float32)
                if self.args.env=="stag_hunt":
                    add_ten[..., self.noop_id] = 1
                elif self.args.env=="sc2":
                    du = th.zeros(self.args.n_actions, device=self.args.device)
                    du[self.noop_id] = 1.
                    add_ten[alive_mask==0] = du
                    du = th.zeros(self.args.n_actions, device=self.args.device)
                    du[1] = 1.
                    add_ten[alive_mask==1] = du
                i_avail_actions = i_avail_actions + (zero_mask.unsqueeze(-1) * add_ten)

                all_zero_ava_after += th.all(i_avail_actions==0, dim=-1).sum()


                i_qs, i_chosen_actions = self.mac.rollout_act(i_avail_actions, i_inputs, t_env=t_env, test_mode=self.args.img_greedy, rollout=self.args.img_greedy)
                mask_i_chosen = i_chosen_actions[:, :-1]

                dead_agent_act += (mask_i_chosen[alive_mask[:, :-1]==0] != self.noop_id).sum()
                img_chosen_acts.append(i_chosen_actions)
                i_onehot_chosen_actions = th.zeros(bs, sl, self.args.n_agents, self.args.n_actions).cuda().scatter_(3, i_chosen_actions.unsqueeze(-1), 1).float()

            i_prior_mu, i_prior_logvar = self.model.prior_encode(i_latent_states, i_onehot_chosen_actions)

            if self.args.rollout_random_scale == 0:
                i_next_latent_states = i_prior_mu
            else:
                eps = th.randn_like(i_prior_logvar) * self.args.rollout_random_scale
                i_next_latent_states = i_prior_mu + eps * th.exp(0.5 * i_prior_logvar).detach()

            i_next_inputs = self.model.decode(i_next_latent_states)
            if self.args.auxiliary_task:
                with th.no_grad():
                    i_avail_actions = self.model.get_avail_action(i_next_latent_states.view(*i_next_latent_states.size()[:2], self.args.n_agents, -1))
                    i_avail_actions = th.distributions.Bernoulli(logits=i_avail_actions).probs
                    i_avail_actions[i_avail_actions >= 0.5] = 1
                    i_avail_actions[i_avail_actions < 0.5] = 0


                    # only noop is available when dead
                    no_op_acts = th.zeros(self.args.n_actions, device=self.args.device)
                    no_op_acts[self.noop_id] = 1
                    i_avail_actions[alive_mask == 0] = no_op_acts

                    # mask out act#1 when agent is alive
                    if self.args.env == "sc2":
                        m = th.ones(self.args.n_actions, device=self.args.device)
                        m[self.noop_id] = 0
                        i_avail_actions[alive_mask != 0] *= m

                    i_avail_actions = i_avail_actions.long()


            i_inputs = i_next_inputs
            i_latent_states = i_next_latent_states
            rollout_states.append(i_prior_mu)

        rollout_states = th.stack(rollout_states, dim=2)
        img_chosen_acts = th.stack(img_chosen_acts, dim=2)
        if self.args.use_combiner:
            rollout_latent_states = self.combiner(batch['state'], rollout_states)
        else:
            traj_encode = self.model.get_traj_encode(rollout_states)
            rollout_latent_states = th.cat([batch['state'], traj_encode], dim=-1)


        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, rollout_latent_states[:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, rollout_latent_states[:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        q_loss = (masked_td_error ** 2).sum() / mask.sum()

        loss = state_loss + q_loss

        if self.args.auxiliary_task:
            loss += avail_actions_loss
        if self.args.use_combiner and self.args.combiner_regular:
            w_reg = self.mse(self.combiner.img_w + self.combiner.s_w,
                             th.ones_like(self.combiner.img_w, device=self.args.device))

            w_reg = w_reg.reshape(bs, sl, -1)[:, :-1]
            loss += w_reg[mask==1].mean() * self.args.reg_beta

        # Optimise
        self.rl_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.rl_optimiser.step()


        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if self.args.z_log_interval!=0 and t_env-self.log_z_t >= self.args.z_log_interval:
            np.save(self.path_post + f'/post_{t_env}.npy', sample_state.detach().cpu().numpy())
            np.save(self.path_prior + f'/prior_{t_env}.npy', next_prior_sample_state.detach().cpu().numpy())
            self.log_z_t = t_env

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            if self.args.validation:
                with th.no_grad():
                    _, _, _, val_obs_mse_loss, val_kld_loss, val_avail_actions_loss, _ = self.model_train(
                        self.validation_batch, self.validation_tau)
                self.logger.log_stat("validation/L_RC_post", val_obs_mse_loss.item(), t_env)
                self.logger.log_stat("validation/KL(q|p)", val_kld_loss.item(), t_env)
                self.logger.log_stat("validation/L_FA", val_avail_actions_loss.item(), t_env)


            if self.args.env=="stag_hunt":
                self.logger.log_stat("mbvd/img_capture", (img_chosen_acts==5).to(th.float32).sum(-1).mean(-1).mean().item(), t_env)

            if self.args.use_combiner:
                self.logger.log_stat("s_w", self.combiner.s_w.reshape(bs, sl, -1)[:, :-1][mask == 1].mean().item(), t_env)
                self.logger.log_stat("img_w",
                                     self.combiner.img_w.reshape(bs, sl, -1)[:, :-1][mask == 1].mean().item(),
                                     t_env)
            self.logger.log_stat("dead_agent_act", dead_agent_act.item(), t_env)
            self.logger.log_stat("all_zero_ava_prev", all_zero_ava_prev.item(), t_env)
            self.logger.log_stat("all_zero_ava_after", all_zero_ava_after.item(), t_env)
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("entropy", entropy.mean().item(), t_env)
            #self.logger.log_stat("mbvd/state_loss", state_loss.item(), t_env)
            self.logger.log_stat("mbvd/L_RC", obs_mse_loss.item(), t_env)
            self.logger.log_stat("mbvd/KL(q|p)", kld_loss.item(),t_env)
            self.logger.log_stat("mbvd/cum_rew_symlog_mse", rew_loss.item(), t_env)
            self.logger.log_stat("rew", rewards[mask == 1].mean().item(),
                                     t_env)
            #self.logger.log_stat("mbvd/prior_loss", prior_loss.item(), t_env)
            # self.logger.log_stat("mbvd/KL(p|N)", prior_kld_loss.item(), t_env)
            # self.logger.log_stat("mbvd/L_RC_prior", (prior_state_mse_loss + prior_action_loss).item(), t_env)
            # for h in range(current_rollout_depth):
            #     self.logger.log_stat(f"policy_kl_{h}", kl_bps[h].item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.model.cuda()
        if self.args.use_combiner:
            self.combiner.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.model.state_dict(), "{}/mbvd.th".format(path))
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.rl_optimiser.state_dict(), "{}/opt.th".format(path))
        if self.args.use_combiner:
            th.save(self.combiner.state_dict(), "{}/combiner.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.model.load_state_dict(th.load("{}/mbvd.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.rl_optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.use_combiner:
            self.combiner.load_state_dict(th.load("{}/combiner.th".format(path), map_location=lambda storage, loc: storage))

    def _build_inputs(self, batch):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, :batch.max_seq_length])  # b1av
        if self.args.obs_last_action:
            inputs.append(th.cat([
                th.zeros_like(batch["actions_onehot"][:, 0]).unsqueeze(1), batch["actions_onehot"][:, :batch.max_seq_length-1]
            ], dim=1))
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.args.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, batch.max_seq_length, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs

    def check_for_nan(self, model):
        for name, param in model.named_parameters():
            if th.isnan(param).any():
                raise ValueError(f"NaN detected in {name}")

    def trainable_params_print(self):
        self.logger.console_logger.info(f"Agent NN params:{sum(param.numel() for param in self.mac.parameters())}")
        self.logger.console_logger.info(f"Mixer NN params:{sum(param.numel() for param in self.mixer.parameters())}")
        self.logger.console_logger.info(f"MBVD params:{sum(param.numel() for param in self.model.parameters())}")
        if self.args.use_combiner:
            self.logger.console_logger.info(f"combiner params:{sum(param.numel() for param in self.combiner.parameters())}")
        self.logger.console_logger.info(f"Optimizer model_opt has {sum(param.numel() for param in self.params)} variables.")

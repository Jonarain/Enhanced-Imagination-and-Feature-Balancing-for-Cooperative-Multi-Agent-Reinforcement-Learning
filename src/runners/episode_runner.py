from envs import REGISTRY as env_REGISTRY
from functools import partial
import numpy as np
from components.episode_buffer import EpisodeBatch


csv_schema = "step, global state, observation, action, reward"

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0   # time steps for one episode

        self.t_env = 0   # time steps for the whole training

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                    "state": [self.env.get_state()],
                    "avail_actions": [self.env.get_avail_actions()],
                    "obs": [self.env.get_obs()],
                }

            cur_alive_agent = [0. if not np.any(i) else 1. for i in self.env.get_obs()]
            pre_transition_data["alive_agent_mask"] = cur_alive_agent

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1

            actions, greedy_mask, entropy = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            reward, terminated, env_info = self.env.step(actions[0])

            episode_return += reward

            post_transition_data = {
                    "actions": actions,
                    "entropy": [(entropy,)],
                    "greedy_mask": greedy_mask,
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }


            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1


        last_avail = self.env.get_avail_actions()
        last_data = {
                "state": [self.env.get_state()],
                "avail_actions": [last_avail],
                "obs": [self.env.get_obs()]
            }
        last_data["alive_agent_mask"] = [0. if not np.any(i) else 1. for i in self.env.get_obs()]
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions, greedy_mask, entropy= self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)
        self.batch.update({"greedy_mask": greedy_mask}, ts=self.t)
        self.batch.update({"entropy": [(entropy,)]}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        update_dict = {k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)}
        cur_stats.update(update_dict)

        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)


        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        # fixed steps periodic log
        if test_mode :
            if (len(self.test_returns) == self.args.test_nepisode):
                self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:  # default: log every 2000 steps
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat("runner_log/"+prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat("runner_log/" + prefix + "return_median", np.median(returns), self.t_env)
        self.logger.log_stat("runner_log/"+prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()


        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat("runner_log/"+prefix + k + "_mean", v/stats["n_episodes"], self.t_env)
        stats.clear()


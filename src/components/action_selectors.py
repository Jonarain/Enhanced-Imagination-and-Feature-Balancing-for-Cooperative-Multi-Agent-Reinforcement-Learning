import numpy as np
import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class CategoricalActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = -float("inf")
        t_softmax = th.nn.Softmax(dim=-1)
        masked_policies = t_softmax(masked_policies)

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions

class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay=self.args.epsilon_decay)
        self.epsilon = self.schedule.eval(0)

        if self.args.agent_only_random_seed != -1:
            self.local_random = np.random.RandomState(self.args.agent_only_random_seed)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, rollout=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        # Returns a tensor with the same size as input that is filled with random numbers
        # from a uniform distribution on the interval [0, 1)

        if rollout:
            picked_actions = masked_q_values.max(dim=-1)[1]
            pick_random = th.zeros(agent_inputs.size()[:-1], device=agent_inputs.device)
        else:
            random_numbers = th.rand(agent_inputs.size()[:-1], device=agent_inputs.device)

            if self.args.agent_only_random_seed != -1:
                random_numbers = th.from_numpy(self.local_random.uniform(size=random_numbers.size())).cuda()

            pick_random = (random_numbers < self.epsilon).long()
            random_actions = Categorical(avail_actions.float()).sample().long()

            picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=-1)[1]
        return picked_actions, 1 - pick_random

class GreedyActionSelector():
    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        picked_actions = masked_q_values.max(dim=2)[1]
        return picked_actions


class RandomActionSelector():
    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        chosen_actions=[]
        aa=avail_actions.squeeze()
        for i in range(self.args.n_agents):
            aai=aa[i,:]
            available_actions = th.nonzero(aai).squeeze(-1)
            if available_actions.numel() > 0:
                action = available_actions[th.randint(0, len(available_actions), (1,))].item()
                chosen_actions.append(action)
            else:
                chosen_actions.append(None)

        return th.Tensor(chosen_actions).unsqueeze(0)


REGISTRY["categorical"] = CategoricalActionSelector
REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
REGISTRY["greedy"] = GreedyActionSelector
REGISTRY["random"] = RandomActionSelector



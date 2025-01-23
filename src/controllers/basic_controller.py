from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        # data storage for one step game
        self.last_individual_q = None
        self.last_state = None
        self.agent_outs = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        if self.args.action_selector!="random":
            if self.args.agent=="rnn_sd":
                agent_outputs= self.forward(ep_batch, t_ep, test_mode=test_mode)[0]
            else:
                agent_outputs= self.forward(ep_batch, t_ep, test_mode=test_mode)
        else:
            agent_outputs=[None]

        chosen_actions, greedy_mask = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        bp = self.local_bp(agent_outputs[bs], avail_actions[bs])
        entropy = th.sum(-bp*th.log(bp+1e-6), dim=-1).mean().item()
        return chosen_actions, greedy_mask, entropy


    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)

        if self.args.agent == "rnn_sd":
            agent_outs, self.hidden_states, local_q = self.agent(agent_inputs, self.hidden_states)
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), local_q
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def local_bp(self, agent_outputs, avail_actions):
        # get boltzmann policy from current local q
        cur_agent_outputs = agent_outputs.clone().detach()
        cur_agent_outputs[avail_actions == 0] = -1e6
        bp = th.nn.functional.softmax(cur_agent_outputs, dim=-1)
        return bp

    def rollout_act(self, avail_actions, inputs, t_env=0, test_mode=True, rollout=True):
        # for mbvd
        # ref: https://proceedings.neurips.cc/paper_files/paper/2022/hash/49be51578b507f37cd8b5fad379af183-Abstract-Conference.html
        agent_outs = self.agent.select(inputs)
        chosen_actions, _ = self.action_selector.select_action(agent_outs, avail_actions, t_env, test_mode=test_mode, rollout=rollout)
        return agent_outs, chosen_actions

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def named_parameters(self):
        return self.agent.named_parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
